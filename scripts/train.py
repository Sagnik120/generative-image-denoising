#!/usr/bin/env python3
"""
Main training entry point. Works identically whether run locally or from
the Colab notebook (notebooks/train_colab.ipynb calls this same function).

Usage (local / Colab cell):
    python scripts/train.py --arch nafnet_unet --data_root /path/to/data
    python scripts/train.py --arch restormer_lite --config custom_config.yaml
    python scripts/train.py --arch pix2pix_gan --epochs 30 --batch_size 8
    python scripts/train.py --arch tiny_ddpm_sr3 --data_root /content/drive/MyDrive/.../data

Only ONE architecture trains per run, by design (per the project's config-
driven structure) -- run this once per architecture you want to compare,
then look at results/<arch>/ for that architecture's full report.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.registry import build_bundle, default_config_path, ARCHITECTURES
from src.common.utils import set_seed, get_device, load_yaml
from src.common.dataset import build_dataloaders
from src.common.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Train one denoising architecture.")
    p.add_argument("--arch", required=True, choices=list(ARCHITECTURES.keys()),
                    help="Which architecture to train.")
    p.add_argument("--config", default=None,
                    help="Path to a config YAML. Defaults to that architecture's own config.yaml.")
    p.add_argument("--data_root", default=str(PROJECT_ROOT / "data"),
                    help="Persistent directory for datasets (use a Google Drive path in Colab).")
    p.add_argument("--results_root", default=str(PROJECT_ROOT / "results"),
                    help="Root results directory (each architecture gets its own subfolder).")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs from config.")
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"],
                    help="Device to train on (defaults to auto-detecting cuda -> mps -> cpu).")
    p.add_argument("--resume", default=None, help="Path to a checkpoint to resume from.")
    return p.parse_args()


def main():
    args = parse_args()
    config_path = args.config or (PROJECT_ROOT / default_config_path(args.arch))
    cfg = load_yaml(config_path)

    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size

    set_seed(cfg["train"].get("seed", 42))
    device = get_device(args.device)
    print(f"[train.py] Architecture: {args.arch}  |  Device: {device}")
    print(f"[train.py] Config: {cfg}")

    train_loader, val_loader = build_dataloaders(
        data_root=args.data_root,
        patch_size=cfg["data"]["patch_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        val_fraction=cfg["data"].get("val_fraction", 0.03),
        download_bsds=cfg["data"].get("download_bsds", True),
        seed=cfg["train"].get("seed", 42),
    )

    bundle = build_bundle(args.arch, cfg, device)

    if args.resume:
        from src.common.utils import load_checkpoint
        print(f"[train.py] Resuming from checkpoint: {args.resume}")
        bundle.load_state_dict(load_checkpoint(args.resume, map_location=device))

    results_dir = Path(args.results_root) / args.arch
    trainer = Trainer(cfg, bundle, train_loader, val_loader, results_dir, args.arch, device)
    trainer.train()


if __name__ == "__main__":
    main()
