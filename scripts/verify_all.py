#!/usr/bin/env python3
"""
Sanity-check script: verifies every architecture builds, runs a forward
pass, a train_step, an eval_step, FLOPs counting, and metric computation --
all on tiny random dummy data, entirely on CPU, in under a minute. Run this
before committing to a long real training run.

Usage:
    python scripts/verify_all.py
"""
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.registry import ARCHITECTURES, build_bundle, default_config_path
from src.common.utils import load_yaml, set_seed
from src.common.metrics import evaluate_batch, compute_flops_and_params
from src.common.degradations import RandomDegradation, ALL_FNS
import numpy as np


def test_degradations():
    print("\n=== Testing degradation pipeline ===")
    img = np.random.rand(64, 64, 3).astype(np.float32)
    deg = RandomDegradation(min_ops=1, max_ops=3)
    for i in range(5):
        out = deg(img)
        assert out.shape == img.shape, f"shape mismatch: {out.shape} vs {img.shape}"
        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0
    print(f"  All {len(ALL_FNS)} individual degradation functions + RandomDegradation OK.")

    for fn in ALL_FNS:
        out = fn(img.copy())
        assert out.shape == img.shape
        assert out.min() >= 0.0 and out.max() <= 1.0
    print("  Each individual degradation function verified independently. PASS")


def test_architecture(arch_name, batch_size=2, patch_size=64):
    print(f"\n=== Testing architecture: {arch_name} ===")
    cfg = load_yaml(PROJECT_ROOT / default_config_path(arch_name))
    # Shrink model + use small patch size for a fast CPU smoke test.
    cfg["train"]["epochs"] = 1
    cfg["data"]["patch_size"] = patch_size

    device = torch.device("cpu")
    set_seed(0)
    bundle = build_bundle(arch_name, cfg, device)

    corrupted = torch.rand(batch_size, 3, patch_size, patch_size)
    clean = torch.rand(batch_size, 3, patch_size, patch_size)

    # train_step
    losses = bundle.train_step(corrupted, clean)
    assert "loss" in losses and np.isfinite(losses["loss"]), f"bad train loss: {losses}"
    print(f"  train_step OK -> {losses}")

    # eval_step
    pred, eval_losses = bundle.eval_step(corrupted, clean)
    assert pred.shape == clean.shape, f"pred shape {pred.shape} != clean shape {clean.shape}"
    assert pred.min() >= 0.0 and pred.max() <= 1.0, "eval_step output not clamped to [0,1]"
    print(f"  eval_step OK -> output shape {tuple(pred.shape)}, losses {eval_losses}")

    # metrics
    m = evaluate_batch(pred, clean, device)
    assert np.isfinite(m["psnr"]) and np.isfinite(m["ssim"])
    print(f"  metrics OK -> PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}  DISTS={m['dists']:.4f}")

    # checkpoint round-trip
    state = bundle.state_dict()
    bundle.load_state_dict(state)
    print("  checkpoint save/load round-trip OK")

    # scheduler step
    if hasattr(bundle, "step_scheduler"):
        bundle.step_scheduler()
    print("  scheduler step OK")

    # FLOPs at the REAL competition resolution (256x256) -- separate quick check
    inference_model = bundle.get_inference_model()
    flop_info = compute_flops_and_params(inference_model, input_size=(1, 3, 256, 256), device=device)
    print(f"  FLOPs @ 256x256 OK -> {flop_info['gflops']:.3f} GFLOPs, "
          f"{flop_info['params']:,} params")

    print(f"  >>> {arch_name}: ALL CHECKS PASSED")
    return True


def main():
    test_degradations()

    results = {}
    for arch_name in ARCHITECTURES:
        try:
            test_architecture(arch_name)
            results[arch_name] = "PASS"
        except Exception:
            print(f"  >>> {arch_name}: FAILED")
            traceback.print_exc()
            results[arch_name] = "FAIL"

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for arch, status in results.items():
        print(f"  {arch:20s} {status}")

    if any(v == "FAIL" for v in results.values()):
        sys.exit(1)
    print("\nAll architectures verified successfully.")


if __name__ == "__main__":
    main()
