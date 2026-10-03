#!/usr/bin/env python3
"""
Post-training evaluation + packaging.

1. Loads an architecture's best checkpoint.
2. Scores it on the fixed benchmark (PSNR / SSIM / DISTS on uint8 images, as
   the brief specifies) and measures FLOPs.
3. Writes results/<arch>/metrics/final_evaluation_summary.json, including
   the breakdown by degradation group, by domain and by individual case, and
   the same breakdown as benchmark_breakdown.csv.
4. Zips results/<arch>/ (without the large training checkpoints unless
   --zip_checkpoints is passed) for copying off the training machine.

Usage:
    python scripts/evaluate.py --arch a07_naf_attn_hybrid --data_root /path/to/data
"""
import argparse
import csv
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.registry import build_bundle, default_config_path, ARCHITECTURES
from src.common.utils import get_device, load_yaml, load_checkpoint, save_json, check_gpu
from src.common.dataset import build_dataloaders
from src.common.metrics import evaluate_batch_per_image, summarize_benchmark, compute_flops_and_params
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
    p.add_argument("--zip_checkpoints", action="store_true",
                    help="Include checkpoints/ in the zip (large).")
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"],
                    help="Device to evaluate on (defaults to auto-detecting cuda -> mps -> cpu).")
    p.add_argument("--allow_big_gpu", action="store_true")
    p.add_argument("--allow_missing_data", action="store_true")
    return p.parse_args()


@torch.no_grad()
def run_full_evaluation(bundle, val_loader, device):
    loss_meter, records = AverageMeter(), []
    for batch in tqdm(val_loader, desc="final evaluation", disable=not sys.stderr.isatty()):
        corrupted, clean = batch[0].to(device), batch[1].to(device)
        pred, losses = bundle.eval_step(corrupted, clean)
        loss_meter.update(losses["loss"], n=corrupted.size(0))
        m = evaluate_batch_per_image(pred, clean, device)
        records += [{"psnr": p, "ssim": s, "dists": d}
                    for p, s, d in zip(m["psnr"], m["ssim"], m["dists"])]
    summary = summarize_benchmark(records, val_loader.dataset.meta)
    overall = summary["overall"]
    return {
        "final_val_loss": loss_meter.avg,
        "final_val_psnr": overall["psnr"],
        "final_val_ssim": overall["ssim"],
        "final_val_dists": overall["dists"],
        "benchmark": summary,
    }


def write_breakdown_csv(summary: dict, path: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "name", "n", "input_psnr", "psnr", "ssim", "dists"])
        w.writerow(["overall", "all", *[summary["overall"][k] for k in
                                         ("n", "input_psnr", "psnr", "ssim", "dists")]])
        for level in ("by_group", "by_domain", "by_case"):
            for name, v in summary[level].items():
                w.writerow([level[3:], name, v["n"], v["input_psnr"], v["psnr"], v["ssim"], v["dists"]])


def main():
    args = parse_args()
    config_path = args.config or (PROJECT_ROOT / default_config_path(args.arch))
    cfg = load_yaml(config_path)
    device = get_device(args.device)
    check_gpu(device, allow_big=args.allow_big_gpu)

    results_dir = Path(args.results_root) / args.arch
    ckpt_path = args.checkpoint or (results_dir / "checkpoints" / "best_model.pt")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train the model first.")

    cfg["data"]["num_workers"] = min(4, cfg["data"]["num_workers"])
    _, val_loader = build_dataloaders(args.data_root, cfg,
                                      allow_missing_data=args.allow_missing_data)

    bundle = build_bundle(args.arch, cfg, device)
    bundle.load_state_dict(load_checkpoint(ckpt_path, map_location=device))
    print(f"[evaluate.py] Loaded checkpoint: {ckpt_path}")

    eval_results = run_full_evaluation(bundle, val_loader, device)
    flop_info = compute_flops_and_params(bundle.get_inference_model(),
                                          input_size=(1, 3, 256, 256), device=device)
    # a04's counter sees one diffusion step; its real cost is `timesteps` passes.
    passes = cfg.get("model", {}).get("timesteps", 1)
    flop_info["gflops"] *= passes
    flop_info["flops"] *= passes

    summary = {
        "architecture": args.arch,
        "checkpoint": str(ckpt_path),
        **eval_results,
        **flop_info,
    }
    save_json(summary, results_dir / "metrics" / "final_evaluation_summary.json")
    write_breakdown_csv(summary["benchmark"], results_dir / "metrics" / "benchmark_breakdown.csv")
    print("[evaluate.py] Final evaluation summary:")
    for k, v in summary.items():
        if k != "benchmark":
            print(f"    {k}: {v}")
    for level in ("by_group", "by_domain"):
        print(f"    {level[3:]:7s} " + "  ".join(
            f"{name}={v['psnr']:.2f}/{v['ssim']:.3f}/{v['dists']:.3f}"
            for name, v in summary["benchmark"][level].items()))

    # ---- Zip results/<arch>, preserving structure ----
    zip_path = Path(args.zip_out or (Path(args.results_root) / f"{args.arch}_results.zip"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(results_dir.rglob("*")):
            rel = path.relative_to(results_dir.parent)
            skip = path.is_dir() or path.name == ".run.lock" or \
                (not args.zip_checkpoints and "checkpoints" in rel.parts)
            if not skip:
                zf.write(path, rel)
    print(f"[evaluate.py] Packaged results into: {zip_path}"
          + ("" if args.zip_checkpoints else "  (checkpoints left out; pass --zip_checkpoints to include)"))


if __name__ == "__main__":
    main()
