import torch
import torch.nn as nn
import json
import os
import time
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer

from models import ProjectionHead, GNNEncoder, GatedFusion
from losses import NTXentLoss
from augmentations import augment
from utils import (set_seed, get_device, setup_logging, stringify_class,
                   CosineWarmupScheduler, ensure_dir)
from config import get_config

logger = setup_logging("fusion")


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Modal Fusion")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--text-model", default=None)
    parser.add_argument("--graph-model", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--model-out", default=None)
    args = parser.parse_args()

    config = get_config(args.config)
    fc = config['fusion']
    tc = config['text_simclr']
    gc = config['graph_simclr']
    ac = config.get('augmentations', {})

    seed = config.get('seed', 42)
    set_seed(seed)
    device = get_device()

    epochs = args.epochs or fc['epochs']
    model_out = args.model_out or fc['model_out']
    text_model_path = args.text_model or tc['model_out']
    graph_model_path = args.graph_model or gc['model_out']
    data_path = args.data or config.get('data', {}).get('structured_json', 'data/structured_ecore.json')

    logger.info(f"Device: {device}")

    logger.info("Loading pre-trained text encoder...")
    sentence_encoder = SentenceTransformer(tc['encoder_name'])

    text_projector = ProjectionHead(
        tc['embedding_dim'], tc['hidden_dim'], tc['projection_dim']
    ).to(device)

    if os.path.exists(text_model_path):
        ckpt = torch.load(text_model_path, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        text_projector.load_state_dict(state)
        logger.info(f"Loaded text projector from {text_model_path}")
    else:
        logger.warning(f"Text model not found at {text_model_path}, using random init")

    text_projector.eval()
    for p in text_projector.parameters():
        p.requires_grad = False

    has_graph = False
    gnn_encoder = None

    if os.path.exists(graph_model_path):
        try:
            gnn_encoder = GNNEncoder(
                gc['node_feature_dim'], gc['hidden_dim'], gc['projection_dim'],
                gc['num_gnn_layers'], gc['dropout']
            ).to(device)
            ckpt = torch.load(graph_model_path, map_location=device, weights_only=False)
            state = ckpt.get('model_state_dict', ckpt)
            gnn_encoder.load_state_dict(state)
            gnn_encoder.eval()
            for p in gnn_encoder.parameters():
                p.requires_grad = False
            has_graph = True
            logger.info(f"Loaded GNN encoder from {graph_model_path}")
        except Exception as e:
            logger.warning(f"Could not load GNN: {e}. Training text-only fusion.")

    if not has_graph:
        logger.info("No graph model available. Using zero-initialized graph embeddings.")

    logger.info(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        models_data = json.load(f)

    all_texts = []
    for model in models_data:
        for cls_obj in model.get('classes', []):
            all_texts.append(stringify_class(cls_obj))

    logger.info(f"Total elements: {len(all_texts)}")

    graph_dim = gc['projection_dim'] * 2
    effective_graph_dim = graph_dim if has_graph else gc['projection_dim'] * 2

    fusion = GatedFusion(
        text_dim=tc['projection_dim'],
        graph_dim=effective_graph_dim,
        fused_dim=fc['fused_dim'],
        gate_hidden=fc['gate_hidden'],
    ).to(device)

    logger.info(f"Fusion params: {sum(p.numel() for p in fusion.parameters()):,}")

    optimizer = torch.optim.AdamW(fusion.parameters(), lr=fc['learning_rate'])
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=1, total_epochs=epochs)
    criterion = NTXentLoss(temperature=fc['temperature'])

    logger.info("Starting fusion training...")
    best_loss = float('inf')

    batch_size = fc['batch_size']

    for epoch in range(epochs):
        fusion.train()
        scheduler.step(epoch)

        indices = np.random.permutation(len(all_texts))
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(indices) - batch_size, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_texts = [all_texts[i] for i in batch_idx]

            aug1 = [augment(t, ac) for t in batch_texts]
            aug2 = [augment(t, ac) for t in batch_texts]

            with torch.no_grad():
                h1 = sentence_encoder.encode(aug1, convert_to_tensor=True).to(device)
                h2 = sentence_encoder.encode(aug2, convert_to_tensor=True).to(device)

                z_text_1 = text_projector(h1)
                z_text_2 = text_projector(h2)

                z_graph_1 = torch.zeros(batch_size, effective_graph_dim, device=device)
                z_graph_2 = torch.zeros(batch_size, effective_graph_dim, device=device)

            fused_1 = fusion(z_text_1, z_graph_1)
            fused_2 = fusion(z_text_2, z_graph_2)

            loss = criterion(fused_1, fused_2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ensure_dir(os.path.dirname(model_out))
            torch.save({
                'epoch': epoch,
                'model_state_dict': fusion.state_dict(),
                'config': fc,
                'has_graph': has_graph,
            }, model_out)

    logger.info(f"Fusion training complete. Best loss: {best_loss:.4f}")

    fusion.eval()
    with torch.no_grad():
        sample_texts = all_texts[:100]
        h = sentence_encoder.encode(sample_texts, convert_to_tensor=True).to(device)
        z_text = text_projector(h)
        z_graph = torch.zeros(len(sample_texts), effective_graph_dim, device=device)
        gates = fusion.get_gate_values(z_text, z_graph)
        avg_gate = gates.mean().item()
        logger.info(f"Average gate value (text weight): {avg_gate:.4f}")
        logger.info("Gate=1.0 means full text reliance, 0.0 means full graph reliance")


if __name__ == "__main__":
    main()
