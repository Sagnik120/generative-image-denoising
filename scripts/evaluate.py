#!/usr/bin/env python3
"""
Post-training evaluation + packaging.

1. Loads an architecture's best checkpoint.
2. Re-runs full validation-set metrics (PSNR / SSIM / DISTS) and FLOPs.
3. Writes a final summary JSON into results/<arch>/metrics/.
4. Zips the ENTIRE results/<arch>/ folder (checkpoints, loss_curves,
   metrics, visualizations, logs) into a single .zip with the exact same
   internal folder structure as the local `results/` folder -- so after
   downloading it from Colab you can unzip it straight into your local
   project's results/ folder with no renaming needed.

Usage:
    python scripts/evaluate.py --arch nafnet_unet --data_root /path/to/data
"""
import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.registry import build_bundle, default_config_path, ARCHITECTURES
from src.common.utils import get_device, load_yaml, load_checkpoint, save_json
from src.common.dataset import build_dataloaders
from src.common.metrics import evaluate_batch, compute_flops_and_params
from src.common.utils import AverageMeter
import torch
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate + package a trained architecture.")
    p.add_argument("--arch", required=True, choices=list(ARCHITECTURES.keys()))
    p.add_argument("--config", default=None)
    p.add_argument("--data_root", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--results_root", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--checkpoint", default=None,
                    help="Defaults to results/<arch>/checkpoints/best_model.pt")
    p.add_argument("--zip_out", default=None,
                    help="Defaults to results/<arch>_results.zip")
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"],
                    help="Device to evaluate on (defaults to auto-detecting cuda -> mps -> cpu).")
    return p.parse_args()


@torch.no_grad()
def run_full_evaluation(bundle, val_loader, device):
    psnr_meter, ssim_meter, dists_meter, loss_meter = (AverageMeter(), AverageMeter(),
                                                         AverageMeter(), AverageMeter())
    for corrupted, clean in tqdm(val_loader, desc="final evaluation"):
        corrupted, clean = corrupted.to(device), clean.to(device)
        pred, losses = bundle.eval_step(corrupted, clean)
        loss_meter.update(losses["loss"], n=corrupted.size(0))
        m = evaluate_batch(pred, clean, device)
        psnr_meter.update(m["psnr"], n=corrupted.size(0))
        ssim_meter.update(m["ssim"], n=corrupted.size(0))
        if m["dists"] == m["dists"]:
            dists_meter.update(m["dists"], n=corrupted.size(0))
    return {
        "final_val_loss": loss_meter.avg,
        "final_val_psnr": psnr_meter.avg,
        "final_val_ssim": ssim_meter.avg,
        "final_val_dists": dists_meter.avg if dists_meter.count > 0 else float("nan"),
    }


def main():
    args = parse_args()
    config_path = args.config or (PROJECT_ROOT / default_config_path(args.arch))
    cfg = load_yaml(config_path)
    device = get_device(args.device)

    results_dir = Path(args.results_root) / args.arch
    ckpt_path = args.checkpoint or (results_dir / "checkpoints" / "best_model.pt")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train the model first.")

    _, val_loader = build_dataloaders(
        data_root=args.data_root,
        patch_size=cfg["data"]["patch_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        val_fraction=cfg["data"].get("val_fraction", 0.03),
        download_bsds=cfg["data"].get("download_bsds", True),
    )

    bundle = build_bundle(args.arch, cfg, device)
    bundle.load_state_dict(load_checkpoint(ckpt_path, map_location=device))
    print(f"[evaluate.py] Loaded checkpoint: {ckpt_path}")

    eval_results = run_full_evaluation(bundle, val_loader, device)
    flop_info = compute_flops_and_params(bundle.get_inference_model(),
                                          input_size=(1, 3, 256, 256), device=device)

    summary = {
        "architecture": args.arch,
        "checkpoint": str(ckpt_path),
        **eval_results,
        **flop_info,
    }
    save_json(summary, results_dir / "metrics" / "final_evaluation_summary.json")
    print("[evaluate.py] Final evaluation summary:")
    for k, v in summary.items():
        print(f"    {k}: {v}")

    # ---- Zip the entire results/<arch> folder, preserving structure ----
    zip_out = args.zip_out or str(Path(args.results_root) / f"{args.arch}_results")
    zip_path = shutil.make_archive(
        base_name=zip_out,        # e.g. results/nafnet_unet_results -> .zip appended
        format="zip",
        root_dir=Path(args.results_root),
        base_dir=args.arch,       # keeps "nafnet_unet/checkpoints/..." as the internal path
    )
    print(f"[evaluate.py] Packaged results into: {zip_path}")
    print(f"[evaluate.py] Unzip this directly into your local project's results/ folder "
          f"(it already matches the results/{args.arch}/... structure).")


if __name__ == "__main__":
    main()
