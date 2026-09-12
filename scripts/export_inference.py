#!/usr/bin/env python3
"""
Exports a SELF-CONTAINED inference package for final submission, matching
the competition's Section 7 requirement: "Trained model weights ... along
with inference code that accepts a 256x256x3 corrupted image and returns
a 256x256x3 denoised image", with output as uint8 [0, 255].

Produces, under results/<arch>/submission/:
    - model_weights.pt        (just the inference model's state_dict)
    - inference.py            (standalone script, only depends on this
                                 project's src/architectures/<arch>/model.py
                                 and torch/numpy/PIL -- no training-only
                                 dependencies like fvcore or DISTS_pytorch)
    - README.md               (usage instructions)

Usage:
    python scripts/export_inference.py --arch nafnet_unet
"""
import argparse
import shutil
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
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]  # results/{arch}/submission -> project root
sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures.{arch}.model import {model_class}

WEIGHTS_PATH = THIS_DIR / "model_weights.pt"

_model = None
_device = None


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
    model = {model_ctor}
    state = torch.load(WEIGHTS_PATH, map_location=_device)
    model.load_state_dict(state)
    model.to(_device).eval()
    _model = model
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

    out = out.clamp(0, 1).squeeze(0).cpu().numpy()
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
- `inference.py` -- standalone script exposing `denoise_image(corrupted_uint8) -> uint8`
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

## Requirements
`torch`, `numpy`, `Pillow` -- deliberately minimal; no training-only
dependencies (fvcore, DISTS_pytorch, tqdm, etc.) are required to run
inference.

## Reported performance (see results/{arch}/metrics/final_evaluation_summary.json)
Run `python scripts/evaluate.py --arch {arch}` from the project root to
(re)generate the latest PSNR / SSIM / DISTS / FLOPs numbers for this
checkpoint.
"""

MODEL_CLASS_MAP = {
    "nafnet_unet": ("NAFNetUNet", "NAFNetUNet()", "out = model(x)"),
    "restormer_lite": ("RestormerLite", "RestormerLite()", "out = model(x)"),
    "pix2pix_gan": ("UNetGenerator", "UNetGenerator()", "out = model(x)"),
    "tiny_ddpm_sr3": ("TinyDDPM", "TinyDDPM()", "out = model.sample(x)"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=list(ARCHITECTURES.keys()))
    p.add_argument("--config", default=None)
    p.add_argument("--results_root", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--checkpoint", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    config_path = args.config or (PROJECT_ROOT / default_config_path(args.arch))
    cfg = load_yaml(config_path)
    device = get_device()

    results_dir = Path(args.results_root) / args.arch
    ckpt_path = args.checkpoint or (results_dir / "checkpoints" / "best_model.pt")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train the model first.")

    bundle = build_bundle(args.arch, cfg, device)
    bundle.load_state_dict(load_checkpoint(ckpt_path, map_location=device))
    inference_model = bundle.get_inference_model()

    submission_dir = results_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    # tiny_ddpm_sr3's get_inference_model() returns a FLOPs-counting wrapper,
    # not the true sampling model -- special-case the saved weights + call.
    if args.arch == "tiny_ddpm_sr3":
        torch.save(bundle.model.state_dict(), submission_dir / "model_weights.pt")
    else:
        torch.save(inference_model.state_dict(), submission_dir / "model_weights.pt")

    model_class, model_ctor, inference_call = MODEL_CLASS_MAP[args.arch]
    script = INFERENCE_TEMPLATE.format(
        arch=args.arch, model_class=model_class, model_ctor=model_ctor,
        inference_call=inference_call,
    )
    (submission_dir / "inference.py").write_text(script)
    (submission_dir / "README.md").write_text(README_TEMPLATE.format(arch=args.arch))

    print(f"[export_inference.py] Submission package written to: {submission_dir}")
    print("  Contents:", [p.name for p in submission_dir.iterdir()])


if __name__ == "__main__":
    main()
