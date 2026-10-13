#!/usr/bin/env python3
"""
Extra evaluation for the round-4 generative models (a20 onward).

scripts/evaluate.py scores a model once, with its default seed and step
count. This script adds what a generative model also has to show:

  - sample diversity: the same benchmark images restored with several noise
    seeds; reported as the mean per-pixel standard deviation in grey levels
    (0-255). Zero means the noise is ignored and the model is, in effect,
    a regression network.
  - for diffusion samplers, PSNR / SSIM / DISTS / FLOPs at several step counts
    from the same checkpoint (--steps 1 2 4).

Usage:
    python scripts/eval_generative.py --arch a22_resdiff --data_root /path/to/data   # steps 1, 2, 4
(scripts/run_all.py calls this automatically for round-4 models.)
    python scripts/eval_generative.py --arch a21_gan_wavelet --data_root /path/to/data

Writes results/<arch>/metrics/generative_eval.json.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.registry import build_bundle, default_config_path, ARCHITECTURES
from src.common.utils import get_device, load_yaml, load_checkpoint, save_json
from src.common.dataset import BenchmarkDataset
from src.common.metrics import evaluate_batch_per_image, compute_flops_and_params
from src.common.gennet import reseed_noise


def parse_args():
    p = argparse.ArgumentParser(description="Diversity and multi-step evaluation of a generative model.")
    p.add_argument("--arch", required=True, choices=list(ARCHITECTURES.keys()))
    p.add_argument("--data_root", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--results_root", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--steps", nargs="*", type=int, default=None,
                   help="Sampling step counts to evaluate (diffusion samplers only).")
    p.add_argument("--seeds", type=int, default=8, help="Noise seeds for the diversity measurement.")
    p.add_argument("--diversity_images", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    return p.parse_args()


@torch.no_grad()
def score(model, loader, device):
    sums, n = {"psnr": 0.0, "ssim": 0.0, "dists": 0.0}, 0
    for batch in loader:
        corrupted, clean = batch[0].to(device), batch[1].to(device)
        pred = model(corrupted).float().clamp(0, 1)
        m = evaluate_batch_per_image(pred, clean, device)
        for k in sums:
            sums[k] += sum(v for v in m[k] if v == v)
        n += corrupted.size(0)
    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def diversity(model, loader, device, seeds, max_images):
    """Mean per-pixel std (grey levels of 255) of the uint8 outputs over noise seeds."""
    total, n = 0.0, 0
    for batch in loader:
        corrupted = batch[0].to(device)
        outs = []
        for seed in range(seeds):
            reseed_noise(model, seed)
            outs.append((model(corrupted).float().clamp(0, 1) * 255.0).round())
        std = torch.stack(outs).std(dim=0)
        total += std.mean(dim=(1, 2, 3)).sum().item()
        n += corrupted.size(0)
        if n >= max_images:
            break
    reseed_noise(model, 0)
    return total / max(n, 1)


def main():
    args = parse_args()
    cfg = load_yaml(PROJECT_ROOT / default_config_path(args.arch))
    device = get_device(args.device)
    results_dir = Path(args.results_root) / args.arch
    ckpt = args.checkpoint or (results_dir / "checkpoints" / "best_model.pt")
    bundle = build_bundle(args.arch, cfg, device)
    bundle.load_state_dict(load_checkpoint(ckpt, map_location=device))
    model = bundle.get_inference_model().eval()
    reseed_noise(model, 0)

    loader = torch.utils.data.DataLoader(BenchmarkDataset(args.data_root), shuffle=False,
                                         batch_size=args.batch_size, num_workers=2)
    has_steps = hasattr(model, "steps")
    default_steps = getattr(model, "steps", 1)
    if has_steps:      # default: 1, 2, 4 and the model's own setting, capped at its training levels
        step_list = args.steps or sorted({1, 2, 4, default_steps})
        step_list = [k for k in step_list if k <= model.timesteps]
    else:
        step_list = [1]

    rows = []
    for k in step_list:
        if has_steps:
            model.steps = k
        row = {"steps": k, **score(model, loader, device)}
        row["diversity"] = diversity(model, loader, device, args.seeds, args.diversity_images)
        row["gflops"] = compute_flops_and_params(model, device=device)["gflops"]
        rows.append(row)
        print(f"[eval_generative] {args.arch} steps={k}: PSNR {row['psnr']:.3f}  "
              f"SSIM {row['ssim']:.4f}  DISTS {row['dists']:.4f}  "
              f"{row['gflops']:.2f} GFLOPs  diversity {row['diversity']:.3f}/255", flush=True)
    if has_steps:
        model.steps = default_steps

    noise_strength = [m.weight.abs().mean().item() for m in model.modules()
                      if m.__class__.__name__ == "NoiseInjection"]
    out = {"architecture": args.arch, "checkpoint": str(ckpt), "seeds": args.seeds,
           "results": rows,
           "mean_noise_strength": sum(noise_strength) / len(noise_strength) if noise_strength else None}
    save_json(out, results_dir / "metrics" / "generative_eval.json")
    print(f"[eval_generative] written to {results_dir / 'metrics' / 'generative_eval.json'}")


if __name__ == "__main__":
    main()
