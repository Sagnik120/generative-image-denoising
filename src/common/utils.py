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
        torch.backends.cudnn.benchmark = True      # fixed 256x256 inputs: let cuDNN tune
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
    """Writes to a temporary file and renames it, so a crash or kill in the
    middle of a save can never leave a truncated checkpoint behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, map_location=None):
    return torch.load(path, map_location=map_location, weights_only=False)


def check_gpu(device, max_gb: float = 30.0, allow_big: bool = False):
    """Prints which GPU this process landed on and refuses to run on one larger
    than `max_gb` (the shared 48 GB card is reserved; GPU numbering differs
    between tools, so the check is on the card itself, not on its index)."""
    if device.type != "cuda":
        return
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024 ** 3
    print(f"[gpu] {props.name}  {total_gb:.1f} GB  "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")
    if total_gb > max_gb and not allow_big:
        raise SystemExit(
            f"[gpu] Refusing to run: this is a {total_gb:.0f} GB GPU, and GPUs above "
            f"{max_gb:.0f} GB are reserved. Pick another one with CUDA_VISIBLE_DEVICES, "
            f"or pass --allow_big_gpu to override.")


def git_commit() -> str:
    import subprocess
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


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
