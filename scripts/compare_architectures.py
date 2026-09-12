#!/usr/bin/env python3
"""
Reads results/<arch>/metrics/final_evaluation_summary.json for every
architecture you've trained + evaluated, and produces a single comparison
table + bar-chart figure so you can pick a winner across all four
competition axes (FLOPs, PSNR, SSIM, DISTS) at a glance.

Usage (after running scripts/evaluate.py for each architecture you trained):
    python scripts/compare_architectures.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.registry import ARCHITECTURES


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", default=str(PROJECT_ROOT / "results"))
    args = p.parse_args()
    results_root = Path(args.results_root)

    rows = []
    for arch in ARCHITECTURES:
        summary_path = results_root / arch / "metrics" / "final_evaluation_summary.json"
        if not summary_path.exists():
            print(f"[compare] Skipping '{arch}' -- no final_evaluation_summary.json found "
                  f"(run scripts/evaluate.py --arch {arch} first).")
            continue
        with open(summary_path) as f:
            data = json.load(f)
        rows.append({
            "architecture": arch,
            "PSNR (dB) [higher better]": round(data.get("final_val_psnr", float("nan")), 3),
            "SSIM [higher better]": round(data.get("final_val_ssim", float("nan")), 4),
            "DISTS [lower better]": round(data.get("final_val_dists", float("nan")), 4),
            "GFLOPs [lower better]": round(data.get("gflops", float("nan")), 3),
            "Params (M)": round(data.get("params", 0) / 1e6, 2),
        })

    if not rows:
        print("[compare] No evaluated architectures found yet. Nothing to compare.")
        return

    df = pd.DataFrame(rows).set_index("architecture")
    out_dir = results_root / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "comparison_table.csv"
    df.to_csv(csv_path)
    print(f"\n[compare] Comparison table saved to {csv_path}\n")
    print(df.to_string())

    # Independent per-axis ranking, mirroring the competition's actual scoring
    # scheme (Section 6): each axis ranked separately, best = rank 1.
    rank_df = pd.DataFrame(index=df.index)
    rank_df["PSNR rank"] = df["PSNR (dB) [higher better]"].rank(ascending=False)
    rank_df["SSIM rank"] = df["SSIM [higher better]"].rank(ascending=False)
    rank_df["DISTS rank"] = df["DISTS [lower better]"].rank(ascending=True)
    rank_df["GFLOPs rank"] = df["GFLOPs [lower better]"].rank(ascending=True)
    rank_df["avg_rank"] = rank_df.mean(axis=1)
    rank_df = rank_df.sort_values("avg_rank")
    rank_csv = out_dir / "comparison_ranking.csv"
    rank_df.to_csv(rank_csv)
    print(f"\n[compare] Per-axis ranking (lower avg_rank = better overall) saved to {rank_csv}\n")
    print(rank_df.to_string())

    # Bar chart dashboard
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    metrics = ["PSNR (dB) [higher better]", "SSIM [higher better]",
               "DISTS [lower better]", "GFLOPs [lower better]"]
    colors = ["#2A9D8F", "#5B5FC7", "#E9A13B", "#E76F51"]
    for ax, metric, color in zip(axes, metrics, colors):
        df[metric].plot(kind="bar", ax=ax, color=color)
        ax.set_title(metric, fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3, axis="y")
    plt.suptitle("Architecture Comparison Across All Four Competition Axes", fontsize=14)
    plt.tight_layout()
    fig_path = out_dir / "comparison_dashboard.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\n[compare] Comparison dashboard figure saved to {fig_path}")


if __name__ == "__main__":
    main()
