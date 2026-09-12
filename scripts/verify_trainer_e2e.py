#!/usr/bin/env python3
"""
End-to-end integration test of the Trainer class using a tiny synthetic
image folder (no internet download needed) -- exercises the full
train -> validate -> log -> checkpoint -> visualize -> final report loop
for one architecture, on CPU, for 2 epochs.

Usage:
    python scripts/verify_trainer_e2e.py --fake_data_dir /tmp/fake_data/train --arch nafnet_unet
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.registry import build_bundle, default_config_path
from src.common.utils import load_yaml, set_seed, get_device
from src.common.dataset import DenoisingDataset, collect_image_paths
from src.common.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="nafnet_unet")
    p.add_argument("--train_dir", default="/tmp/fake_data/train")
    p.add_argument("--val_dir", default="/tmp/fake_data/val")
    p.add_argument("--results_root", default="/tmp/fake_results")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(PROJECT_ROOT / default_config_path(args.arch))
    cfg["train"]["epochs"] = 2
    cfg["data"]["patch_size"] = 64
    cfg["data"]["batch_size"] = 2
    cfg["data"]["num_workers"] = 0
    cfg["train"]["visualize_every"] = 1

    set_seed(0)
    device = torch.device("cpu")

    train_paths = collect_image_paths({"train": args.train_dir})
    val_paths = collect_image_paths({"val": args.val_dir})
    print(f"train_paths={len(train_paths)}  val_paths={len(val_paths)}")

    train_ds = DenoisingDataset(train_paths, patch_size=64, augment=True, degrade=True)
    val_ds = DenoisingDataset(val_paths, patch_size=64, augment=False, degrade=True)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    bundle = build_bundle(args.arch, cfg, device)
    results_dir = Path(args.results_root) / args.arch
    trainer = Trainer(cfg, bundle, train_loader, val_loader, results_dir, args.arch, device)
    trainer.train()

    # Verify expected artifacts exist
    expected = [
        results_dir / "logs" / "training_log.csv",
        results_dir / "checkpoints" / "best_model.pt",
        results_dir / "checkpoints" / "last_model.pt",
        results_dir / "loss_curves" / "loss_curve.png",
        results_dir / "metrics" / "final_report.json",
        results_dir / "visualizations" / "epoch_0001_comparison.png",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        print("MISSING EXPECTED ARTIFACTS:")
        for m in missing:
            print(" -", m)
        sys.exit(1)

    print("\nAll expected artifacts were created successfully:")
    for p in expected:
        print(" -", p)
    print(f"\n[verify_trainer_e2e] {args.arch}: END-TO-END TEST PASSED")


if __name__ == "__main__":
    main()
