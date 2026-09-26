#!/usr/bin/env python3
"""
End-to-end rehearsal of the full chain for each architecture, on CPU:

    train (a few steps, real data loader, validation on the benchmark,
           checkpoint, resume, final report)
      -> evaluate (benchmark breakdown + FLOPs + zip)
      -> export   (self-contained package, re-loaded in a clean process)

It needs a prepared data root (scripts/prepare_data.py). Outputs go to
<results_root>/_dry_run/, never to real results. On the GPU server the same
rehearsal is `python scripts/run_all.py --data_root ... --dry_run`.

Usage:
    python scripts/verify_trainer_e2e.py --data_root /path/to/data
    python scripts/verify_trainer_e2e.py --data_root ... --archs a05_nafnet_v2 --steps 3
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.registry import ARCHITECTURES, ROUND2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--results_root", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--archs", nargs="*", default=ROUND2, choices=list(ARCHITECTURES))
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--device", default="cpu", choices=["cuda", "mps", "cpu"])
    p.add_argument("--allow_missing_data", action="store_true")
    args = p.parse_args()

    dry_root = str(Path(args.results_root) / "_dry_run")
    extra = ["--allow_missing_data"] if args.allow_missing_data else []
    results = {}
    for arch in args.archs:
        base = [sys.executable]
        chain = [
            ("train", ["scripts/train.py", "--arch", arch, "--data_root", args.data_root,
                       "--results_root", args.results_root, "--dry_run",
                       "--dry_run_steps", str(args.steps), "--device", args.device,
                       "--batch_size", "2", "--num_workers", "0", *extra]),
            ("evaluate", ["scripts/evaluate.py", "--arch", arch, "--data_root", args.data_root,
                          "--results_root", dry_root, "--device", args.device, *extra]),
            ("export", ["scripts/export_inference.py", "--arch", arch,
                        "--results_root", dry_root, "--device", args.device]),
        ]
        results[arch] = "PASS"
        for name, cmd in chain:
            print(f"\n=== {arch}: {name} ===", flush=True)
            if subprocess.call(base + cmd, cwd=str(PROJECT_ROOT)) != 0:
                results[arch] = f"FAIL at {name}"
                break

        expected = [Path(dry_root) / arch / rel for rel in (
            "logs/training_log.csv", "checkpoints/best_model.pt", "checkpoints/last_model.pt",
            "metrics/final_report.json", "metrics/final_evaluation_summary.json",
            "metrics/benchmark_breakdown.csv", "visualizations/epoch_0001_comparison.png",
            "submission/model_weights.pt", "submission/inference.py")]
        missing = [str(e) for e in expected if not e.exists()]
        if results[arch] == "PASS" and missing:
            results[arch] = f"FAIL: missing {missing}"

    print("\n" + "=" * 60 + "\nEND-TO-END SUMMARY\n" + "=" * 60)
    for arch, status in results.items():
        print(f"  {arch:26s} {status}")
    if any(v != "PASS" for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
