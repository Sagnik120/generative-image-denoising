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
| `training/` | full training and evaluation code, to retrain the model from scratch (see below) |

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

## Training from scratch (`training/`)

`model.py` + `inference.py` are a standalone copy of the generator for inference. `training/` holds the code that trained it:
- data download and preparation;
- the corruption pipeline;
- generator and discriminator;
- losses;
- the training loop;
- evaluation.

```
training/
├── requirements.txt
├── scripts/
│   ├── prepare_data.py      # downloads/caches the clean datasets, builds the fixed benchmarks
│   ├── train.py             # trains the model
│   ├── evaluate.py          # PSNR / SSIM / DISTS / FLOPs on the benchmarks (also per case, group, domain)
│   └── eval_generative.py   # sample diversity and quality at several test-time noise scales
└── src/
    ├── registry.py
    ├── architectures/generative_denoiser/   # model entry point + config.yaml (all hyperparameters)
    └── common/              # degradations.py, dataset.py, gennet.py (generator), gan.py (discriminator,
                             # adversarial + diversity training), losses.py, trainer.py, metrics.py, ...
```

Commands (from `training/`, one 24 GB GPU; the full run takes ~21–25 h):
```bash
pip install -r requirements.txt
python scripts/prepare_data.py --data_root <DATA>                      # datasets + benchmarks
python scripts/train.py --arch generative_denoiser --data_root <DATA>   # 200k steps, writes results/generative_denoiser/
python scripts/evaluate.py --arch generative_denoiser --data_root <DATA> --noise_scale 0.75
python scripts/eval_generative.py --arch generative_denoiser --data_root <DATA>
```
All hyperparameters are in `src/architectures/generative_denoiser/config.yaml`. The trained generator weights (`checkpoints/best_model.pt`, key `ema`) are what `model_weights.pt` contains.

## Results on our held-out benchmark
396 images (natural, low-light, medical), 33 corruption cases (single, pairs, triples); uint8 images, skimage PSNR/SSIM, official DISTS-pytorch.

| Split | PSNR (dB) | SSIM | DISTS |
|---|---|---|---|
| All 396 images (input: 24.49 dB) | **32.74** | **0.874** | **0.137** |
| Natural | 29.25 | 0.838 | 0.141 |
| Low-light | 37.73 | 0.901 | 0.161 |
| Medical | 34.74 | 0.919 | 0.105 |
| Corruption types never seen in training (96 images) | 32.21 | 0.830 | 0.178 |

All at the submitted noise scale 0.75.
