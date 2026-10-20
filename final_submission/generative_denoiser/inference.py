"""
Inference for the generative denoiser.

Python:
    from inference import denoise_image
    clean = denoise_image(corrupted)          # (256, 256, 3) uint8 -> (256, 256, 3) uint8

Command line, one image:
    python inference.py --input corrupted.png --output denoised.png

Command line, a folder of images (every image in it, same file names in the output folder):
    python inference.py --input corrupted_dir/ --output denoised_dir/

`load_model()` returns the torch.nn.Module used for inference (input/output:
float tensor (1, 3, 256, 256) in [0, 1]); pass it to fvcore / ptflops to
measure FLOPs. The noise is drawn inside its forward pass.
"""
import argparse
import json
import sys
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
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

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


def read_rgb(path):
    """Reads any image as a (H, W, 3) uint8 RGB array (greyscale/RGBA are converted)."""
    return np.array(Image.open(path).convert("RGB"))


def save_image(array, path):
    """Saves a uint8 RGB array. PNG/BMP/TIFF are lossless; JPEG/WebP outputs are
    saved at the highest quality (they stay lossy, so prefer PNG inputs or --png)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        Image.fromarray(array).save(path, quality=100, subsampling=0)
    elif ext == ".webp":
        Image.fromarray(array).save(path, lossless=True)
    else:
        Image.fromarray(array).save(path)


def denoise_file(src, dst, seed=None):
    image = read_rgb(src)
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"{src}: expected a 256x256 RGB image, got {image.shape[1]}x{image.shape[0]}")
    out = denoise_image(image, seed=seed)
    save_image(out, dst)
    return out


def denoise_folder(src_dir, dst_dir, seed=None, force_png=False):
    """Denoises every image in src_dir (not recursive) into dst_dir with the same
    file name (with --png the extension becomes .png). Returns (done, failed)."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise SystemExit(f"no images ({', '.join(sorted(IMAGE_EXTENSIONS))}) found in {src_dir}")
    done, failed = 0, []
    for i, src in enumerate(files, 1):
        dst = dst_dir / (src.stem + ".png" if force_png else src.name)
        try:
            denoise_file(src, dst, seed=seed)
            done += 1
        except Exception as e:                      # keep going; report at the end
            failed.append((src.name, str(e)))
        if i % 50 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] {src.name} -> {dst}", flush=True)
    print(f"done: {done} image(s) written to {dst_dir}" + (f"; {len(failed)} failed" if failed else ""))
    for name, err in failed:
        print(f"  FAILED {name}: {err}", file=sys.stderr)
    return done, failed


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Denoise one 256x256 RGB image or a folder of them.")
    p.add_argument("--input", required=True, help="an image file, or a folder of images")
    p.add_argument("--output", required=True,
                   help="output image file (for a file input) or output folder (for a folder input)")
    p.add_argument("--seed", type=int, default=None, help="fixed noise seed for a reproducible output")
    p.add_argument("--png", action="store_true",
                   help="folder mode: always write .png (same name, lossless) instead of the input's extension")
    args = p.parse_args()
    src = Path(args.input)
    if src.is_dir():
        _, failed = denoise_folder(src, args.output, seed=args.seed, force_png=args.png)
        sys.exit(1 if failed else 0)
    denoise_file(src, args.output, seed=args.seed)
    print(f"saved {args.output}")
