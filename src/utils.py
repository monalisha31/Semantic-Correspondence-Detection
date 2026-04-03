import torch
import numpy as np
import random
import os
import json
import logging
from typing import Dict, List, Optional


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(name: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s %(name)s %(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def stringify_class(cls_obj: dict) -> str:
    name = cls_obj.get('name', 'UnnamedClass')

    supertypes = cls_obj.get('supertypes', [])
    supertypes_str = ", ".join(supertypes) if supertypes else "None"

    attributes = []
    for attr in cls_obj.get('attributes', []):
        attributes.append(f"{attr['name']} ({attr['type']})")
    attributes_str = ", ".join(attributes) if attributes else "None"

    references = []
    for ref in cls_obj.get('references', []):
        c_str = "Composition" if ref.get('containment') else "Association"
        mult = ref.get('multiplicity', '')
        mult_str = f" [{mult}]" if mult else ""
        references.append(f"{ref['name']} -> {ref['type']}{mult_str} [{c_str}]")
    references_str = ", ".join(references) if references else "None"

    return (
        f"Class: {name}. "
        f"Supertypes: {supertypes_str}. "
        f"Attributes: {attributes_str}. "
        f"References: {references_str}."
    )


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(data, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


class CosineWarmupScheduler:

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1.0 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * factor)

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]
