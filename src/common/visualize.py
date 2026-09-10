"""
Visualization helpers: loss curves, metric curves, and qualitative
side-by-side (corrupted | prediction | ground truth) comparison grids.
Every architecture's Trainer calls these, writing outputs into that
architecture's results/<arch_name>/{loss_curves,visualizations}/ folders.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .metrics import tensor_to_uint8


def plot_loss_curves(csv_path: str, out_dir: str, arch_name: str = ""):
    df = pd.read_csv(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "train_loss" in df.columns and "val_loss" in df.columns:
        plt.figure(figsize=(7, 4.5))
        plt.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2)
        plt.plot(df["epoch"], df["val_loss"], label="Val Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{arch_name} — Training / Validation Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png", dpi=150)
        plt.close()

    if "gen_loss" in df.columns and "disc_loss" in df.columns:
        plt.figure(figsize=(7, 4.5))
        plt.plot(df["epoch"], df["gen_loss"], label="Generator Loss", linewidth=2)
        plt.plot(df["epoch"], df["disc_loss"], label="Discriminator Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{arch_name} — GAN Loss Curves")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "gan_loss_curve.png", dpi=150)
        plt.close()


def plot_metric_curves(csv_path: str, out_dir: str, arch_name: str = ""):
    df = pd.read_csv(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_specs = [
        ("val_psnr", "PSNR (dB) — higher is better"),
        ("val_ssim", "SSIM — higher is better"),
        ("val_dists", "DISTS — lower is better"),
    ]
    for col, title in metric_specs:
        if col not in df.columns:
            continue
        plt.figure(figsize=(7, 4.5))
        plt.plot(df["epoch"], df[col], linewidth=2, color="#12314F")
        plt.xlabel("Epoch")
        plt.ylabel(col)
        plt.title(f"{arch_name} — {title}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{col}_curve.png", dpi=150)
        plt.close()

    # Combined dashboard
    present = [c for c, _ in metric_specs if c in df.columns]
    if present:
        fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4))
        if len(present) == 1:
            axes = [axes]
        for ax, col in zip(axes, present):
            ax.plot(df["epoch"], df[col], linewidth=2, color="#12314F")
            ax.set_title(col)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)
        plt.suptitle(f"{arch_name} — Validation Metrics Dashboard")
        plt.tight_layout()
        plt.savefig(out_dir / "metrics_dashboard.png", dpi=150)
        plt.close()


def save_comparison_grid(corrupted: torch.Tensor, prediction: torch.Tensor,
                          clean: torch.Tensor, out_path: str, n: int = 6,
                          epoch: int = None, arch_name: str = ""):
    """corrupted/prediction/clean: (B, C, H, W) tensors in [0, 1]."""
    n = min(n, corrupted.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]

    col_titles = ["Corrupted (input)", "Model Output", "Ground Truth (clean)"]
    for row in range(n):
        imgs = [tensor_to_uint8(corrupted[row]), tensor_to_uint8(prediction[row]),
                tensor_to_uint8(clean[row])]
        for col in range(3):
            axes[row, col].imshow(imgs[col])
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=11)

    title = f"{arch_name} — Sample Predictions"
    if epoch is not None:
        title += f" (epoch {epoch})"
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def save_error_heatmap(prediction: torch.Tensor, clean: torch.Tensor, out_path: str, n: int = 4):
    """Visualizes |pred - clean| per pixel, useful to see WHERE the model struggles."""
    n = min(n, prediction.shape[0])
    fig, axes = plt.subplots(n, 1, figsize=(4, 3 * n))
    if n == 1:
        axes = [axes]
    for row in range(n):
        pred = prediction[row].detach().clamp(0, 1).cpu().numpy()
        gt = clean[row].detach().clamp(0, 1).cpu().numpy()
        err = np.abs(pred - gt).mean(axis=0)
        im = axes[row].imshow(err, cmap="inferno", vmin=0, vmax=0.3)
        axes[row].axis("off")
        axes[row].set_title(f"|pred - gt| sample {row}", fontsize=10)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()
