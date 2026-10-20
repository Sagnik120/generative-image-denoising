# Generative Denoiser: a small generative model for image denoising

A conditional GAN generator that removes noise, blur, compression and small artifacts from 256×256 RGB images (natural, low-light and medical). One forward pass, **4.98 GFLOPs**, 5.52 M parameters.

## Input / output
- **Input:** one corrupted RGB image, `numpy.ndarray` of shape **(256, 256, 3)**, dtype **uint8** (0–255).
- **Output:** the denoised RGB image, shape **(256, 256, 3)**, dtype **uint8** (0–255). The network's float output is clamped to [0, 1], scaled by 255, rounded and converted to uint8 inside `denoise_image`.

## Usage
```bash
pip install -r requirements.txt
```
```python
import numpy as np
from PIL import Image
from inference import denoise_image

corrupted = np.array(Image.open("corrupted.png").convert("RGB"))   # (256, 256, 3) uint8
denoised = denoise_image(corrupted)                                 # (256, 256, 3) uint8
Image.fromarray(denoised).save("denoised.png")
```
Command line:
```bash
python inference.py --input corrupted.png --output denoised.png
```
Run from inside this folder (or add it to `PYTHONPATH`). A GPU is used if available; on CPU one image takes well under a second.

**Generative sampling.** The generator injects Gaussian noise inside the network, so each call returns a new sample of the restored image (mean per-pixel standard deviation between samples ≈ 0.8 grey levels). For a reproducible output pass a seed: `denoise_image(corrupted, seed=0)` or `--seed 0`.

## Files
| File | Content |
|---|---|
| `model_weights.pt` | trained generator weights, PyTorch `state_dict` (exponential moving average of the training weights) |
| `config.json` | network size settings |
| `model.py` | the network (NAF U-Net with gated noise injection in the decoder) |
| `inference.py` | `load_model()`, `denoise_image()`, command line |
| `requirements.txt` | `torch`, `numpy`, `Pillow`; `fvcore` / `ptflops` only to measure FLOPs |

## Model
- **Generator:** U-Net of NAFNet blocks (Chen et al., ECCV 2022). Width 24; encoder 2/2/2/4 blocks at 256/128/64/32 px; 3 blocks at 16 px; decoder 1/1/1/2 blocks; the output is a correction added to the input image.
- **Stochasticity:** after each of the 5 decoder blocks, one channel of Gaussian noise is added, scaled by a learned per-channel strength and a learned per-pixel gate. At inference the noise is multiplied by 0.75 (`NOISE_SCALE` in `inference.py`).
- **Training:** adversarially, with a conditional U-Net discriminator on wavelet detail bands, together with reconstruction losses (Charbonnier, FFT, SSIM, DISTS, wavelet) and a sample-diversity constraint. Details are in the technical report.

## FLOPs
Measured on a (1, 3, 256, 256) input, one forward pass (the noise is generated inside the call):

| Counter | Value |
|---|---|
| fvcore | **4.98 GFLOPs** |
| ptflops | 5.09 GMACs |
| Parameters | 5.52 M |

```python
import torch
from fvcore.nn import FlopCountAnalysis
from inference import load_model
print(FlopCountAnalysis(load_model(), torch.rand(1, 3, 256, 256)).total() / 1e9, "GFLOPs")
```

## Results on our held-out benchmark
396 images (natural, low-light, medical), 33 corruption cases (single, pairs, triples); uint8 images, skimage PSNR/SSIM, official DISTS-pytorch.

| PSNR (dB) | SSIM | DISTS |
|---|---|---|
| 32.74 | 0.874 | 0.137 |
