"""General-purpose utilities shared across all architectures."""
import os
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def get_device(preferred: str = None):
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(1, self.count)


class CSVLogger:
    """Append-only CSV logger for per-epoch metrics -- used to draw loss/metric curves later."""

    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def log(self, row: dict):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.t0


def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location=None):
    return torch.load(path, map_location=map_location)


def save_json(obj: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
