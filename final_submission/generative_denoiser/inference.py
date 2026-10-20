"""
Inference for the generative denoiser.

    from inference import denoise_image
    clean = denoise_image(corrupted)          # (256, 256, 3) uint8 -> (256, 256, 3) uint8

Command line:
    python inference.py --input corrupted.png --output denoised.png [--seed 0]

`load_model()` returns the torch.nn.Module used for inference (input/output:
float tensor (1, 3, 256, 256) in [0, 1]); pass it to fvcore / ptflops to
measure FLOPs. The noise is drawn inside its forward pass.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import Generator, set_noise_scale, set_noise_seed

THIS_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = THIS_DIR / "model_weights.pt"
CONFIG_PATH = THIS_DIR / "config.json"
NOISE_SCALE = 0.75          # test-time noise multiplier used for the submission
IMAGE_SIZE = 256

_model, _device = None, None


def load_model(device="cpu"):
    """The generator in eval mode, with the submission's noise scale applied."""
    cfg = json.loads(CONFIG_PATH.read_text())
    model = Generator(**cfg)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    set_noise_scale(model, NOISE_SCALE)
    return model.to(device).eval()


def _get_model():
    global _model, _device
    if _model is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _device = torch.device("mps")
        else:
            _device = torch.device("cpu")
        _model = load_model(_device)
    return _model, _device


@torch.no_grad()
def denoise_image(corrupted, seed=None):
    """corrupted: (256, 256, 3) RGB image, uint8 in [0, 255].
    Returns the denoised (256, 256, 3) RGB image, uint8 in [0, 255].
    seed=None draws new noise (a new sample) on every call; an int gives a
    reproducible sample."""
    img = np.asarray(corrupted)
    assert img.shape == (IMAGE_SIZE, IMAGE_SIZE, 3), f"expected (256, 256, 3), got {img.shape}"
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    model, device = _get_model()
    set_noise_seed(model, seed)
    x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    out = model(x).float().clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
    return (out * 255.0).round().astype(np.uint8)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Denoise one 256x256 RGB image.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    image = np.array(Image.open(args.input).convert("RGB"))
    Image.fromarray(denoise_image(image, seed=args.seed)).save(args.output)
    print(f"saved {args.output}")
