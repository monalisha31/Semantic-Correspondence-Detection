import torch
import torch.nn as nn
import os
import json
import time
import argparse
import random
import numpy as np
from sentence_transformers import SentenceTransformer

from models import GNNEncoder
from losses import NTXentLoss
from utils import (set_seed, get_device, setup_logging, stringify_class,
                   CosineWarmupScheduler, ensure_dir)
from config import get_config

logger = setup_logging("graph_simclr")


def build_graphs_from_structured(data_path: str, encoder: SentenceTransformer):
    try:
        from torch_geometric.data import Data
    except ImportError:
        logger.error("torch_geometric required for graph training. Install it first.")
        return []

    with open(data_path, 'r') as f:
        models_data = json.load(f)

    graphs = []

    for model in models_data:
        classes = model.get('classes', [])
        if len(classes) < 2:
            continue

        name_to_idx = {}
        texts = []
        for i, cls in enumerate(classes):
            name_to_idx[cls['name']] = i
            texts.append(stringify_class(cls))

        with torch.no_grad():
            x = encoder.encode(texts, convert_to_tensor=True)

        edge_list = []
        for cls in classes:
            src = name_to_idx.get(cls['name'])
            if src is None:
                continue

            for st in cls.get('supertypes', []):
                tgt = name_to_idx.get(st)
                if tgt is not None:
                    edge_list.append([src, tgt])
                    edge_list.append([tgt, src])

            for ref in cls.get('references', []):
                tgt = name_to_idx.get(ref['type'])
                if tgt is not None:
                    edge_list.append([src, tgt])
                    edge_list.append([tgt, src])

        if not edge_list:
            edge_list = [[i, i] for i in range(len(classes))]

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        data = Data(x=x, edge_index=edge_index)
        data.model_id = model.get('id', 'unknown')
        data.num_classes = len(classes)

        graphs.append(data)

    return graphs


def augment_graph(data, edge_drop_rate: float = 0.2,
                  feature_mask_rate: float = 0.2):
    data_aug = data.clone()

    if data_aug.edge_index.size(1) > 0:
        keep_mask = torch.rand(data_aug.edge_index.size(1)) > edge_drop_rate
        if keep_mask.any():
            data_aug.edge_index = data_aug.edge_index[:, keep_mask]

    if data_aug.x.size(0) > 0:
        node_mask = torch.rand(data_aug.x.size(0)) > feature_mask_rate
        data_aug.x = data_aug.x.clone()
        data_aug.x[~node_mask] = 0.0

    return data_aug


class GraphPairDataset(torch.utils.data.Dataset):

    def __init__(self, graphs, edge_drop_rate=0.2, feature_mask_rate=0.2):
        self.graphs = graphs
        self.edge_drop_rate = edge_drop_rate
        self.feature_mask_rate = feature_mask_rate

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        v1 = augment_graph(g, self.edge_drop_rate, self.feature_mask_rate)
        v2 = augment_graph(g, self.edge_drop_rate, self.feature_mask_rate)
        return v1, v2


def collate_pair(batch):
    try:
        from torch_geometric.data import Batch
    except ImportError:
        raise ImportError("torch_geometric required")

    v1_list = [pair[0] for pair in batch]
    v2_list = [pair[1] for pair in batch]

    return Batch.from_data_list(v1_list), Batch.from_data_list(v2_list)


def main():
    parser = argparse.ArgumentParser(description="Train Graph SimCLR")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-out", default=None)
    args = parser.parse_args()

    config = get_config(args.config)
    gc = config['graph_simclr']

    seed = config.get('seed', 42)
    set_seed(seed)
    device = get_device()

    epochs = args.epochs or gc['epochs']
    data_path = args.data or config.get('data', {}).get('structured_json', 'data/structured_ecore.json')
    model_out = args.model_out or gc['model_out']

    logger.info(f"Device: {device}, Seed: {seed}")

    logger.info("Building graphs from structured data...")
    sentence_encoder = SentenceTransformer(config['text_simclr']['encoder_name'])
    graphs = build_graphs_from_structured(data_path, sentence_encoder)

    if args.limit:
        graphs = graphs[:args.limit]

    logger.info(f"Built {len(graphs)} graphs")

    if not graphs:
        logger.error("No graphs built. Exiting.")
        return

    avg_nodes = np.mean([g.num_nodes for g in graphs])
    avg_edges = np.mean([g.edge_index.size(1) for g in graphs])
    logger.info(f"Avg nodes: {avg_nodes:.1f}, Avg edges: {avg_edges:.1f}")

    dataset = GraphPairDataset(
        graphs,
        edge_drop_rate=gc['edge_drop_rate'],
        feature_mask_rate=gc['feature_mask_rate'],
    )

    try:
        from torch_geometric.data import DataLoader as PyGDataLoader
        dataloader = PyGDataLoader(
            dataset, batch_size=gc['batch_size'], shuffle=True,
            drop_last=True, collate_fn=collate_pair,
        )
    except ImportError:
        logger.error("torch_geometric DataLoader required. Install torch_geometric.")
        return

    model = GNNEncoder(
        input_dim=gc['node_feature_dim'],
        hidden_dim=gc['hidden_dim'],
        output_dim=gc['projection_dim'],
        num_layers=gc['num_gnn_layers'],
        dropout=gc['dropout'],
    ).to(device)

    logger.info(f"GNN params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=gc['learning_rate'], weight_decay=gc['weight_decay']
    )
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=2, total_epochs=epochs)
    criterion = NTXentLoss(temperature=gc['temperature'])

    logger.info("Starting graph SimCLR training...")
    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        scheduler.step(epoch)
        total_loss = 0.0
        n_batches = 0

        for batch_v1, batch_v2 in dataloader:
            batch_v1 = batch_v1.to(device)
            batch_v2 = batch_v2.to(device)

            _, z1 = model(batch_v1.x, batch_v1.edge_index, batch_v1.batch)
            _, z2 = model(batch_v2.x, batch_v2.edge_index, batch_v2.batch)

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss={avg_loss:.4f} | LR={scheduler.get_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ensure_dir(os.path.dirname(model_out))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': gc,
            }, model_out)

    elapsed = time.time() - start_time
    logger.info(f"Graph training complete in {elapsed:.1f}s. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
