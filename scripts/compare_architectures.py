#!/usr/bin/env python3
"""
Reads results/<arch>/metrics/final_evaluation_summary.json for every
architecture you've trained + evaluated, and produces a single comparison
table + bar-chart figure so you can pick a winner across all four
competition axes (FLOPs, PSNR, SSIM, DISTS) at a glance.

Only architectures scored on the fixed benchmark (round 2 onward) are
compared with each other; round 1 used a different, noisy validation set, so
its numbers are listed in results/INDEX.md for reference but not ranked.

Outputs, under results/comparison/:
    comparison_table.csv        the four axes per architecture
    comparison_ranking.csv      per-axis ranks (the competition's scoring scheme)
    comparison_by_group.csv     PSNR / SSIM / DISTS per degradation group and domain
    comparison_by_case.csv      PSNR per individual degradation / combination
    comparison_dashboard.png
and results/INDEX.md, the numbered list of every architecture so far.

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

from src.registry import ARCHITECTURES, ROUND1


def write_index(results_root: Path, summaries: dict):
    """results/INDEX.md: every architecture in number order, newest last."""
    lines = ["# Architecture index", "",
             "Numbered in the order the architectures were added; the highest number is the newest.",
             "Round 1 was scored on a different validation set and is not comparable with later rounds.",
             "", "| # | Architecture | Round | PSNR (dB) | SSIM | DISTS | GFLOPs | Status |",
             "|---|---|---|---|---|---|---|---|"]
    for arch in ARCHITECTURES:
        d = summaries.get(arch)
        rnd = "1 (old pipeline)" if arch in ROUND1 else "2+ (fixed benchmark)"
        if d is None:
            lines.append(f"| {arch[1:3]} | `{arch}` | {rnd} | - | - | - | - | not run yet |")
        else:
            lines.append(f"| {arch[1:3]} | `{arch}` | {rnd} | {d['final_val_psnr']:.2f} | "
                         f"{d['final_val_ssim']:.4f} | {d['final_val_dists']:.4f} | "
                         f"{d['gflops']:.2f} | evaluated |")
    (results_root / "INDEX.md").write_text("\n".join(lines) + "\n")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", default=str(PROJECT_ROOT / "results"))
    args = p.parse_args()
    results_root = Path(args.results_root)

    rows, summaries = [], {}
    for arch in ARCHITECTURES:
        summary_path = results_root / arch / "metrics" / "final_evaluation_summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            data = json.load(f)
        if arch == "a04_tiny_ddpm_sr3" and "benchmark" not in data:
            data["gflops"] *= 8          # round-1 summary stored one of its 8 diffusion steps
        summaries[arch] = data
        if "benchmark" not in data:      # round 1: not scored on the fixed benchmark
            continue
        rows.append({
            "architecture": arch,
            "PSNR (dB) [higher better]": round(data.get("final_val_psnr", float("nan")), 3),
            "SSIM [higher better]": round(data.get("final_val_ssim", float("nan")), 4),
            "DISTS [lower better]": round(data.get("final_val_dists", float("nan")), 4),
            "GFLOPs [lower better]": round(data.get("gflops", float("nan")), 3),
            "Params (M)": round(data.get("params", 0) / 1e6, 2),
        })

    write_index(results_root, summaries)
    print(f"[compare] Architecture index written to {results_root / 'INDEX.md'}")

    if not rows:
        print("[compare] No architecture evaluated on the fixed benchmark yet. Nothing to compare.")
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

    # Where each model is strong or weak: by degradation group / domain / case.
    group_rows, case_rows = [], []
    for arch in df.index:
        bench = summaries[arch]["benchmark"]
        row = {"architecture": arch}
        for level in ("by_group", "by_domain"):
            for name, v in bench[level].items():
                row[f"{name} PSNR"] = v["psnr"]
                row[f"{name} SSIM"] = v["ssim"]
                row[f"{name} DISTS"] = v["dists"]
        for name, v in summaries[arch].get("benchmark_v2", {}).get("by_group", {}).items():
            row[f"{name} PSNR"], row[f"{name} SSIM"], row[f"{name} DISTS"] = v["psnr"], v["ssim"], v["dists"]
        group_rows.append(row)
        case_rows.append({"architecture": arch,
                          **{name: v["psnr"] for name, v in bench["by_case"].items()}})
    group_df = pd.DataFrame(group_rows).set_index("architecture")
    group_df.to_csv(out_dir / "comparison_by_group.csv")
    case_df = pd.DataFrame(case_rows).set_index("architecture").T
    case_df.insert(0, "input", pd.Series(
        {name: v["input_psnr"] for name, v in summaries[df.index[0]]["benchmark"]["by_case"].items()}))
    case_df.to_csv(out_dir / "comparison_by_case.csv")
    print("\n[compare] PSNR by degradation group and domain:\n")
    print(group_df[[c for c in group_df.columns if c.endswith("PSNR")]].to_string())
    print("\n[compare] PSNR by case (rows) -- 'input' is the corrupted image itself:\n")
    print(case_df.round(2).to_string())

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
    plt.close(fig)
    print(f"\n[compare] Comparison dashboard figure saved to {fig_path}")


if __name__ == "__main__":
    main()
