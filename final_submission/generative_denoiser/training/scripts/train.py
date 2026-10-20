#!/usr/bin/env python3
"""
Main training entry point. Works identically whether run locally or from
the Colab notebook (notebooks/train_colab.ipynb calls this same function).

Usage:
    python scripts/train.py --arch a07_naf_attn_hybrid --data_root /path/to/data
    python scripts/train.py --arch a05_nafnet_v2 --data_root ... --dry_run     # 30-step rehearsal
    python scripts/train.py --arch a05_nafnet_v2 --data_root ... --resume auto # continue if interrupted

On a multi-GPU machine choose the GPU from outside, e.g.
    CUDA_VISIBLE_DEVICES=1 nohup python scripts/train.py --arch ... > a05.log 2>&1 &

Only ONE architecture trains per run, by design (per the project's config-
driven structure). To train several, one after another on each free GPU,
use scripts/run_all.py.
"""
import argparse
import fcntl
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.registry import build_bundle, default_config_path, ARCHITECTURES
from src.common.utils import (set_seed, get_device, load_yaml, load_checkpoint, save_json,
                              check_gpu, git_commit)
from src.common.dataset import build_dataloaders
from src.common.trainer import Trainer

DRY_RUN_STEPS = 30


def parse_args():
    p = argparse.ArgumentParser(description="Train one denoising architecture.")
    p.add_argument("--arch", required=True, choices=list(ARCHITECTURES.keys()),
                    help="Which architecture to train.")
    p.add_argument("--config", default=None,
                    help="Path to a config YAML. Defaults to that architecture's own config.yaml.")
    p.add_argument("--data_root", default=str(PROJECT_ROOT / "data"),
                    help="Directory prepared by scripts/prepare_data.py.")
    p.add_argument("--results_root", default=str(PROJECT_ROOT / "results"),
                    help="Root results directory (each architecture gets its own subfolder).")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs from config.")
    p.add_argument("--batch_size", type=int, default=None, help="Override batch size from config.")
    p.add_argument("--num_workers", type=int, default=None, help="Override data workers from config.")
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"],
                    help="Device to train on (defaults to auto-detecting cuda -> mps -> cpu).")
    p.add_argument("--resume", default=None,
                    help="Checkpoint path, or 'auto' to continue from results/<arch>/checkpoints/"
                         "last_model.pt when it exists (and start fresh when it does not).")
    p.add_argument("--dry_run", action="store_true",
                    help=f"Rehearsal: {DRY_RUN_STEPS} real training steps, a short validation, a "
                         f"checkpoint and the final report, written to results/_dry_run/<arch>. "
                         f"Reports speed and peak GPU memory. Run before every long run.")
    p.add_argument("--dry_run_steps", type=int, default=DRY_RUN_STEPS,
                    help="Training steps in a dry run.")
    p.add_argument("--allow_missing_data", action="store_true",
                    help="Smoke tests only: train even if some dataset views are missing.")
    p.add_argument("--allow_big_gpu", action="store_true",
                    help="Allow running on a GPU larger than 30 GB (reserved by default).")
    return p.parse_args()


def main():
    args = parse_args()
    config_path = args.config or (PROJECT_ROOT / default_config_path(args.arch))
    cfg = load_yaml(config_path)

    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.dry_run:
        cfg["train"]["epochs"] = 1
        cfg["train"]["visualize_every"] = 1
        if isinstance(cfg["train"].get("loss"), dict):
            # Switch the DISTS loss on from step 0, so the speed and memory
            # figures of the rehearsal include it (it normally starts mid-run).
            cfg["train"]["loss"]["dists_start_frac"] = 0.0

    set_seed(cfg["train"].get("seed", 42))
    device = get_device(args.device)
    check_gpu(device, allow_big=args.allow_big_gpu)
    print(f"[train.py] Architecture: {args.arch}  |  Device: {device}  |  commit: {git_commit()}")
    print(f"[train.py] Config: {cfg}")

    results_root = Path(args.results_root) / "_dry_run" if args.dry_run else Path(args.results_root)
    results_dir = results_root / args.arch
    results_dir.mkdir(parents=True, exist_ok=True)

    # One training process per architecture: a second one started by mistake
    # would overwrite the first one's checkpoints and logs.
    lock = open(results_dir / ".run.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"[train.py] Another process is already training {args.arch} "
                         f"into {results_dir}. Not starting a second one.")

    train_loader, val_loader = build_dataloaders(args.data_root, cfg,
                                                 allow_missing_data=args.allow_missing_data)
    bundle = build_bundle(args.arch, cfg, device)
    trainer = Trainer(cfg, bundle, train_loader, val_loader, results_dir, args.arch, device,
                      dry_run_steps=args.dry_run_steps if args.dry_run else 0)

    resume_path = args.resume
    if resume_path == "auto":
        resume_path = trainer.last_ckpt_path if trainer.last_ckpt_path.exists() else None
    if resume_path and not args.dry_run:
        print(f"[train.py] Resuming from checkpoint: {resume_path}")
        state = load_checkpoint(resume_path, map_location=device)
        bundle.load_state_dict(state)
        trainer.resume(state.get("trainer"))
        # A resumed process must not replay the crops and corruptions of epoch 1.
        set_seed(cfg["train"].get("seed", 42) + trainer.start_epoch)

    save_json({"arch": args.arch, "commit": git_commit(), "config": cfg, "argv": sys.argv},
              results_dir / "logs" / "run_config.json")
    trainer.train()

    if args.dry_run:
        print(f"[train.py] DRY RUN OK for {args.arch}: training, validation, checkpointing and "
              f"the final report all ran. See the it/s and gpu_mem figures above.")


if __name__ == "__main__":
    main()
