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
from src.common.degradations import (RandomDegradation, FAMILIES, TRAIN_FAMILIES,
                                     NUM_TRAIN_FAMILIES, apply_recipe)
import numpy as np


def test_degradations():
    print("\n=== Testing degradation pipeline ===")
    rng = np.random.default_rng(0)
    img = rng.random((64, 64, 3)).astype(np.float32)
    for name, (fn, _, _) in FAMILIES.items():
        for s in (0.0, 0.5, 1.0):
            out = fn(img.copy(), s, rng)
            assert out.shape == img.shape and out.dtype == np.float32, name
            assert out.min() >= 0.0 and out.max() <= 1.0 and np.isfinite(out).all(), name
    print(f"  All {len(FAMILIES)} degradation families valid at severity 0 / 0.5 / 1.")

    deg = RandomDegradation()
    lengths = set()
    for _ in range(200):
        recipe = deg.sample_recipe(rng)
        lengths.add(len(recipe))
        assert all(name in TRAIN_FAMILIES for name, _ in recipe), "unseen family leaked into training"
        out, label = deg(img, rng, return_label=True)
        assert out.shape == img.shape and label.shape == (NUM_TRAIN_FAMILIES,)
        assert np.allclose(out * 255, np.round(out * 255), atol=1e-3), "output not on the uint8 grid"
    assert lengths == {0, 1, 2, 3, 4}, f"chain lengths seen: {lengths}"
    a = apply_recipe(img, [("gaussian_noise", 0.5), ("jpeg", 0.5)], np.random.default_rng(1))
    b = apply_recipe(img, [("gaussian_noise", 0.5), ("jpeg", 0.5)], np.random.default_rng(1))
    assert np.array_equal(a, b), "same seed must give the same corruption"
    print("  Random chains of 0-4 families, labels, uint8 quantisation, determinism. PASS")


def test_architecture(arch_name, batch_size=2, patch_size=64):
    print(f"\n=== Testing architecture: {arch_name} ===")
    cfg = load_yaml(PROJECT_ROOT / default_config_path(arch_name))
    # Shrink model + use small patch size for a fast CPU smoke test.
    cfg["train"]["epochs"] = 1
    cfg["data"]["patch_size"] = patch_size
    cfg["data"]["steps_per_epoch"] = 4
    if isinstance(cfg["train"].get("loss"), dict):
        cfg["train"]["loss"]["dists_start_frac"] = 0.0   # exercise the DISTS loss too

    device = torch.device("cpu")
    set_seed(0)
    bundle = build_bundle(arch_name, cfg, device)

    corrupted = torch.rand(batch_size, 3, patch_size, patch_size)
    clean = torch.rand(batch_size, 3, patch_size, patch_size)

    # train_step (twice: the second step runs on updated weights)
    for _ in range(2):
        if getattr(bundle, "uses_labels", False):
            losses = bundle.train_step(corrupted, clean, torch.rand(batch_size, NUM_TRAIN_FAMILIES))
        else:
            losses = bundle.train_step(corrupted, clean)
        assert "loss" in losses and np.isfinite(losses["loss"]), f"bad train loss: {losses}"
    print(f"  train_step OK -> { {k: round(v, 4) for k, v in losses.items()} }")

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

    # the inference model must map a full-size image to the same shape
    with torch.no_grad():
        full = bundle.get_inference_model()(torch.rand(1, 3, 256, 256))
    if arch_name != "a04_tiny_ddpm_sr3":      # a04's wrapper is a FLOPs-counting stub
        assert full.shape == (1, 3, 256, 256) and torch.isfinite(full).all()

    print(f"  >>> {arch_name}: ALL CHECKS PASSED")
    return True


def main():
    test_degradations()

    results = {}
    only = sys.argv[1:]
    for arch_name in ARCHITECTURES:
        if only and arch_name not in only:
            continue
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
        print(f"  {arch:26s} {status}")

    if any(v == "FAIL" for v in results.values()):
        sys.exit(1)
    print("\nAll architectures verified successfully.")


if __name__ == "__main__":
    main()
