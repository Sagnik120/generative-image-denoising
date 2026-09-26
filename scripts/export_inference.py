#!/usr/bin/env python3
"""
Exports a SELF-CONTAINED inference package for final submission, matching
the competition's Section 7 requirement: "Trained model weights ... along
with inference code that accepts a 256x256x3 corrupted image and returns
a 256x256x3 denoised image", with output as uint8 [0, 255].

Produces, under results/<arch>/submission/:
    - model_weights.pt        (the inference model's state_dict; EMA weights
                                 for round-2 architectures)
    - model_config.json       (round 2: the `model:` section used to rebuild it)
    - src/                    (a copy of the model code, so the folder runs
                                 on its own, away from this repository)
    - inference.py            (denoise_image() for images, load_model() for
                                 FLOPs measurement)
    - README.md               (usage instructions)

After writing the package it is loaded in a fresh Python process and run on
a random image, and the result is compared with the training-side model, so
a broken export is caught here rather than by the organisers.

Usage:
    python scripts/export_inference.py --arch a07_naf_attn_hybrid
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.registry import build_bundle, default_config_path, ARCHITECTURES
from src.common.utils import get_device, load_yaml, load_checkpoint
import torch


INFERENCE_TEMPLATE = '''"""
Standalone inference script for the "{arch}" denoising submission.

Usage:
    from inference import denoise_image
    clean = denoise_image(corrupted_uint8_array)   # both (256, 256, 3) uint8

Or from the command line:
    python inference.py --input corrupted.png --output denoised.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))   # the package carries its own copy of src/

from src.architectures.{arch}.model import {model_class}

WEIGHTS_PATH = THIS_DIR / "model_weights.pt"

_model = None
_device = None


def load_model(device="cpu"):
    """Returns the network as an nn.Module in eval mode. It maps a float tensor
    (1, 3, 256, 256) in [0, 1] to the same shape -- pass it to fvcore / ptflops
    to measure FLOPs."""
    model = {model_ctor}
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


def _load_model():
    global _model, _device
    if _model is not None:
        return _model, _device
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")
    _model = load_model(_device)
    return _model, _device


@torch.no_grad()
def denoise_image(corrupted_uint8: np.ndarray) -> np.ndarray:
    """corrupted_uint8: (256, 256, 3) uint8 RGB array. Returns (256, 256, 3) uint8 RGB."""
    assert corrupted_uint8.shape == (256, 256, 3), \\
        f"Expected (256, 256, 3), got {{corrupted_uint8.shape}}"
    model, device = _load_model()

    x = corrupted_uint8.astype(np.float32) / 255.0
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)

    {inference_call}

    out = out.float().clamp(0, 1).squeeze(0).cpu().numpy()
    out = (out.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    img = np.array(Image.open(args.input).convert("RGB"))
    result = denoise_image(img)
    Image.fromarray(result).save(args.output)
    print(f"Saved denoised image to {{args.output}}")
'''

README_TEMPLATE = """# Submission Package -- {arch}

This folder is a self-contained inference package for the "{arch}"
architecture, ready for the competition's Section 7 submission requirement.

## Contents
- `model_weights.pt` -- trained weights (inference model only, no optimizer state)
- `model_config.json` -- the model's size settings (round-2 architectures)
- `src/` -- the model code; this folder runs on its own
- `inference.py` -- `denoise_image(corrupted_uint8) -> uint8` and `load_model()`
- `README.md` -- this file

## Usage
```python
from inference import denoise_image
import numpy as np
from PIL import Image

corrupted = np.array(Image.open("corrupted.png").convert("RGB"))  # (256, 256, 3) uint8
clean = denoise_image(corrupted)                                   # (256, 256, 3) uint8
Image.fromarray(clean).save("denoised.png")
```

Or from the command line:
```bash
python inference.py --input corrupted.png --output denoised.png
```

## Measuring FLOPs
```python
import torch
from fvcore.nn import FlopCountAnalysis
from inference import load_model

model = load_model()                                  # nn.Module, eval mode
print(FlopCountAnalysis(model, torch.rand(1, 3, 256, 256)).total() / 1e9, "GFLOPs")
```

## Requirements
`torch`, `numpy`, `Pillow`, `opencv-python-headless` -- no training-only
dependencies (fvcore, DISTS_pytorch, tqdm, etc.) are required to run
inference.

## Reported performance (see results/{arch}/metrics/final_evaluation_summary.json)
Run `python scripts/evaluate.py --arch {arch}` from the project root to
(re)generate the latest PSNR / SSIM / DISTS / FLOPs numbers for this
checkpoint.
"""

# Round-1 architectures are rebuilt from their class defaults. Round-2
# architectures (anything not listed here) expose build_model(model_cfg) and
# are rebuilt from the saved model_config.json.
MODEL_CLASS_MAP = {
    "a01_nafnet_unet": ("NAFNetUNet", "NAFNetUNet()", "out = model(x)"),
    "a02_restormer_lite": ("RestormerLite", "RestormerLite()", "out = model(x)"),
    "a03_pix2pix_gan": ("UNetGenerator", "UNetGenerator()", "out = model(x)"),
    "a04_tiny_ddpm_sr3": ("TinyDDPM", "TinyDDPM()", "out = model.sample(x)"),
}
ROUND2_ENTRY = ("build_model",
                'build_model(json.loads((THIS_DIR / "model_config.json").read_text()))',
                "out = model(x)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=list(ARCHITECTURES.keys()))
    p.add_argument("--config", default=None)
    p.add_argument("--results_root", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    return p.parse_args()


def copy_model_code(submission_dir: Path, arch: str):
    """Copies src/common and the architecture's folder (plus any other
    architecture folder its model.py imports) into the package."""
    dst = submission_dir / "src"
    if dst.exists():
        shutil.rmtree(dst)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(PROJECT_ROOT / "src" / "common", dst / "common", ignore=ignore)
    (dst / "__init__.py").write_text("")
    (dst / "architectures").mkdir()
    (dst / "architectures" / "__init__.py").write_text("")
    needed, queue = set(), [arch]
    while queue:
        name = queue.pop()
        if name in needed:
            continue
        needed.add(name)
        source = (PROJECT_ROOT / "src" / "architectures" / name / "model.py").read_text()
        queue += [other for other in ARCHITECTURES if f"src.architectures.{other}." in source]
    for name in needed:
        shutil.copytree(PROJECT_ROOT / "src" / "architectures" / name,
                        dst / "architectures" / name, ignore=ignore)


VERIFY_SNIPPET = """
import sys, numpy as np, torch
sys.path.insert(0, {sub!r})
import inference
img = np.load({inp!r})
out = inference.denoise_image(img)
assert out.shape == (256, 256, 3) and out.dtype == np.uint8, (out.shape, out.dtype)
np.save({outp!r}, out)
m = inference.load_model()
assert m(torch.rand(1, 3, 256, 256)).shape == (1, 3, 256, 256)
"""


def verify_package(submission_dir: Path, reference_model, device, sample_call):
    """Runs the exported package in a clean process and checks it reproduces the
    training-side model's output on the same random image."""
    import numpy as np
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    inp, outp = submission_dir / "_verify_in.npy", submission_dir / "_verify_out.npy"
    np.save(inp, img)
    try:
        subprocess.run([sys.executable, "-c", VERIFY_SNIPPET.format(
            sub=str(submission_dir), inp=str(inp), outp=str(outp))],
            check=True, cwd=str(submission_dir))
        exported = np.load(outp)
    finally:
        inp.unlink(missing_ok=True)
        outp.unlink(missing_ok=True)

    if sample_call is None:          # stochastic sampler (a04): shape/dtype check only
        return None
    with torch.no_grad():
        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        ref = sample_call(reference_model, x).float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    ref = (ref * 255.0).round().astype(np.uint8)
    return int(np.abs(ref.astype(int) - exported.astype(int)).max())


def main():
    args = parse_args()
    config_path = args.config or (PROJECT_ROOT / default_config_path(args.arch))
    cfg = load_yaml(config_path)
    device = get_device(args.device)

    results_dir = Path(args.results_root) / args.arch
    ckpt_path = args.checkpoint or (results_dir / "checkpoints" / "best_model.pt")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train the model first.")

    bundle = build_bundle(args.arch, cfg, device)
    bundle.load_state_dict(load_checkpoint(ckpt_path, map_location=device))
    inference_model = bundle.get_inference_model()

    submission_dir = results_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    # a04_tiny_ddpm_sr3's get_inference_model() returns a FLOPs-counting wrapper,
    # not the true sampling model -- special-case the saved weights + call.
    if args.arch == "a04_tiny_ddpm_sr3":
        inference_model = bundle.model
    inference_model = inference_model.eval()
    torch.save(inference_model.state_dict(), submission_dir / "model_weights.pt")

    if args.arch in MODEL_CLASS_MAP:
        model_class, model_ctor, inference_call = MODEL_CLASS_MAP[args.arch]
    else:
        model_class, model_ctor, inference_call = ROUND2_ENTRY
        (submission_dir / "model_config.json").write_text(json.dumps(cfg.get("model", {}), indent=2))

    script = INFERENCE_TEMPLATE.format(
        arch=args.arch, model_class=model_class, model_ctor=model_ctor,
        inference_call=inference_call,
    )
    (submission_dir / "inference.py").write_text(script)
    (submission_dir / "README.md").write_text(README_TEMPLATE.format(arch=args.arch))
    copy_model_code(submission_dir, args.arch)

    sample_call = None if args.arch == "a04_tiny_ddpm_sr3" else (lambda m, x: m(x))
    max_diff = verify_package(submission_dir, inference_model, device, sample_call)
    print(f"[export_inference.py] Submission package written to: {submission_dir}")
    print("  Contents:", sorted(p.name for p in submission_dir.iterdir()))
    if max_diff is None:
        print("  Verified: package loads and returns a (256, 256, 3) uint8 image.")
    else:
        print(f"  Verified in a clean process: uint8 output, max difference from the "
              f"training-side model = {max_diff} grey level(s).")
        if max_diff > 2:
            raise SystemExit("[export_inference.py] ERROR: exported package does not reproduce "
                             "the trained model's output.")


if __name__ == "__main__":
    main()
