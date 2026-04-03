import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import time
import argparse

from models import ProjectionHead
from losses import NTXentLoss, HardNegativeNTXentLoss
from augmentations import augment
from utils import (set_seed, get_device, setup_logging, stringify_class,
                   CosineWarmupScheduler, ensure_dir)
from config import get_config

logger = setup_logging("text_simclr")


class TextSimCLRDataset(Dataset):

    def __init__(self, texts: list, aug_config: dict = None):
        self.texts = texts
        self.aug_config = aug_config or {}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        v1 = augment(text, self.aug_config)
        v2 = augment(text, self.aug_config)
        return v1, v2


def train_epoch(encoder, projector, dataloader, optimizer, criterion,
                device, epoch: int):
    projector.train()
    total_loss = 0.0
    total_pos_sim = 0.0
    n_batches = 0

    for batch_idx, (b_v1, b_v2) in enumerate(dataloader):
        with torch.no_grad():
            h_i = encoder.encode(list(b_v1), convert_to_tensor=True).to(device).clone()
            h_j = encoder.encode(list(b_v2), convert_to_tensor=True).to(device).clone()

        z_i = projector(h_i)
        z_j = projector(h_j)

        loss = criterion(z_i, z_j)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)

        optimizer.step()

        with torch.no_grad():
            z_i_n = nn.functional.normalize(z_i, dim=1)
            z_j_n = nn.functional.normalize(z_j, dim=1)
            pos_sim = (z_i_n * z_j_n).sum(dim=1).mean().item()

        total_loss += loss.item()
        total_pos_sim += pos_sim
        n_batches += 1

        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"  Epoch {epoch+1} Batch {batch_idx+1}/{len(dataloader)} "
                f"Loss={loss.item():.4f} PosSim={pos_sim:.4f}"
            )

    return total_loss / n_batches, total_pos_sim / n_batches


def validate(encoder, projector, dataloader, criterion, device):
    projector.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for b_v1, b_v2 in dataloader:
            h_i = encoder.encode(list(b_v1), convert_to_tensor=True).to(device)
            h_j = encoder.encode(list(b_v2), convert_to_tensor=True).to(device)
            z_i = projector(h_i)
            z_j = projector(h_j)
            loss = criterion(z_i, z_j)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(1, n_batches)


def main():
    parser = argparse.ArgumentParser(description="Train Text SimCLR")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-out", default=None)
    args = parser.parse_args()

    config = get_config(args.config)
    tc = config['text_simclr']
    ac = config.get('augmentations', {})

    seed = config.get('seed', 42)
    set_seed(seed)
    device = get_device()

    epochs = args.epochs or tc['epochs']
    data_path = args.data or config.get('data', {}).get('structured_json', 'data/structured_ecore.json')
    model_out = args.model_out or tc['model_out']

    logger.info(f"Device: {device}, Seed: {seed}")
    logger.info(f"Config: epochs={epochs}, batch_size={tc['batch_size']}, "
                f"lr={tc['learning_rate']}, temp={tc['temperature']}")

    logger.info(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        models_data = json.load(f)

    if args.limit:
        models_data = models_data[:args.limit]

    all_texts = []
    for model in models_data:
        for cls_obj in model.get('classes', []):
            all_texts.append(stringify_class(cls_obj))

    logger.info(f"Total class descriptions: {len(all_texts)}")

    if not all_texts:
        logger.error("No data loaded. Exiting.")
        return

    n_val = max(1, int(len(all_texts) * 0.1))
    val_texts = all_texts[:n_val]
    train_texts = all_texts[n_val:]

    logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    train_ds = TextSimCLRDataset(train_texts, ac)
    val_ds = TextSimCLRDataset(val_texts, ac)

    train_dl = DataLoader(train_ds, batch_size=tc['batch_size'], shuffle=True,
                          drop_last=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=tc['batch_size'], shuffle=False,
                        drop_last=True, num_workers=0)

    encoder = SentenceTransformer(tc['encoder_name'])
    projector = ProjectionHead(
        input_dim=tc['embedding_dim'],
        hidden_dim=tc['hidden_dim'],
        output_dim=tc['projection_dim'],
        dropout=0.1,
    ).to(device)

    logger.info(f"Projector params: {sum(p.numel() for p in projector.parameters()):,}")

    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=tc['learning_rate'],
        weight_decay=tc['weight_decay'],
    )

    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=tc.get('warmup_epochs', 2),
        total_epochs=epochs,
    )

    criterion = NTXentLoss(temperature=tc['temperature'])

    logger.info("Starting training...")
    best_val_loss = float('inf')
    start_time = time.time()

    history = {'train_loss': [], 'val_loss': [], 'pos_sim': [], 'lr': []}

    for epoch in range(epochs):
        scheduler.step(epoch)
        current_lr = scheduler.get_lr()[0]

        train_loss, pos_sim = train_epoch(
            encoder, projector, train_dl, optimizer, criterion, device, epoch
        )
        val_loss = validate(encoder, projector, val_dl, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['pos_sim'].append(pos_sim)
        history['lr'].append(current_lr)

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | "
            f"PosSim={pos_sim:.4f} | LR={current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ensure_dir(os.path.dirname(model_out))
            torch.save({
                'epoch': epoch,
                'model_state_dict': projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': tc,
            }, model_out)
            logger.info(f"  Saved best model (val_loss={val_loss:.4f})")

    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed:.1f}s")
    logger.info(f"Best val loss: {best_val_loss:.4f}")

    history_path = model_out.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"History saved to {history_path}")


if __name__ == "__main__":
    main()
