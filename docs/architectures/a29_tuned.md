# a29_tuned: a29 with Optuna trial 32 hyperparameters (FINAL MODEL)

Round 8. Tuning: `../HYPERPARAMETER_TUNING.md`. Plan: `../PLAN.md`.

**How it works:** identical network and losses to a29 (`a29_gan_decnoise.md`).
- **Generator:** conditional GAN with a NAF U-Net (width 24; encoder [2,2,2,4] at 256/128/64/32 px; bottleneck 3 blocks at 16 px; decoder [1,1,1,2]; residual output). Noise is injected only after the 5 decoder blocks, multiplied by a learned per-channel strength and a learned per-pixel gate (sigmoid of a 1×1 conv). **4.98 GFLOPs (fvcore), 5.52 M parameters, one pass.**
- **Discriminator** (training only): conditional U-Net with spectral norm on the Haar detail bands of candidate and corrupted image.
- **Losses:**
  - composite (Charbonnier 1.0 + FFT 0.05 + (1 − SSIM) **0.334** + DISTS 0.1 from 50%) and wavelet L1 0.05, on every output;
  - diversity hinge (target **1.60** grey levels, weight 4.0) on 4 extra noise draws per batch;
  - hinge adversarial **0.00275** (5k ramp); discriminator hinge + R1.
- **Optimiser:** AdamW, lr **7.55e-4** (a29: 4e-4), discriminator lr = **0.5** × generator lr, cosine to 1e-6, 1k warm-up, EMA 0.999, AMP.
- **Data:** v3 (26 corruption families, 0–4 per image, 12% clean), 16 images + 4 extra draws per step, **200k steps (80 epochs)**, from scratch.

## Results
Benchmark v1 (396 images, 33 cases), v2 (96, unseen families), v3 check set (96). Run time 25.4 h. Stable curve: best checkpoint = last epoch.

| Noise scale | PSNR | SSIM | DISTS | Diversity |
|---|---|---|---|---|
| ×0.5 | 32.864 | 0.8746 | 0.1381 | 0.548 (< 0.6) |
| ×0.6 | 32.817 | 0.8744 | 0.1377 | 0.640 |
| **×0.75 (submission setting)** | **32.741** | **0.8740** | **0.1371** | **0.772** |
| ×1 (as trained) | 32.613 | 0.8733 | 0.1364 | 0.974 |
| ×1.5 | 32.373 | 0.8717 | 0.1367 | 1.310 |
| a29 ×0.5 (before tuning) | 32.269 | 0.8682 | 0.1360 | 0.86 |
| a19 (regression, not allowed) | 33.112 | 0.8743 | 0.1293 | 0 |

| Group (×1) | single | extra | pair | triple | unseen | natural | lowlight | medical | unseen_v2 | plan | v3 check |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PSNR | 35.64 | 35.56 | 31.01 | 29.75 | 32.00 | 29.16 | 37.64 | 34.49 | **32.15** | 36.29 | 33.97 |
| vs a29 | +1.01 | +1.16 | +0.49 | +0.37 | +0.60 | +0.46 | +1.16 | +0.67 | **+1.24** | +0.84 | +0.52 |

| Epoch | 12 | 20 | 30 | 40 | 50 | 60 | 70 | 80 |
|---|---|---|---|---|---|---|---|---|
| val PSNR (×1) | 31.30 | 31.76 | 32.12 | 32.37 | 32.48 | 32.57 | 32.61 | 32.61 |
| a29 | 30.97 | 31.46 | 31.80 | 32.08 | – | 31.85 | 31.91 | 31.93 |
| `train_spread255` | 1.85 | 1.81 | 1.76 | 1.75 | 1.77 | 1.73 | 1.70 | 1.69 |

## Analysis
- **Tuning paid: +0.69 dB at ×1 over a29 (+0.47 dB at the eligible scales), +0.008 SSIM.** The 10k-step Optuna gain (+0.5 dB) carried over to 200k, and grew: unlike a29, PSNR kept rising after the DISTS loss started (32.37 at epoch 40 → 32.61 at 80; a29 fell from 32.16 to 31.93). The higher learning rate and lower adversarial weight let the model keep improving fidelity while DISTS improved.
- **Close to the regression reference:** at ×0.75 it is −0.37 dB, −0.0003 SSIM and +0.008 DISTS from a19, which is not allowed (a29: −0.84 dB). SSIM is practically equal.
- **Generalisation improved most:** `unseen_v2` 32.15, **above a19 (31.88)** and every other model; `line_dropout` 39.91 (a19 38.97), `film_grain` 28.48. Low-light +1.16 dB over a29.
- **Structured cases recovered:** `small_artifacts` 42.28 (a29 39.36, a25_full 40.59), `salt_pepper` 40.84 (a29 38.16), `periodic_noise` 37.70, `chromatic_aberration` 38.86. Still below a19 there (45.46 / 41.40 / 39.91 / 40.55).
- **Remaining weakness:** `resize_blur` 33.86 is still below its input (34.24); `jpeg` +0.59 over the input (a19 +1.29).
- **DISTS is the one metric not improved** vs a29 (0.1371 at ×0.75 vs a29 0.1360 at ×0.5).
- **Diversity:** 0.97 at ×1, 0.77 at ×0.75 (chosen for a safe margin over 0.6), 0.64 at ×0.6, 0.55 at ×0.5. Tuning lowered diversity at equal scale vs a29 (×0.5: 0.55 vs 0.86).

**Verdict: final submission model, at noise ×0.75: 32.74 / 0.8740 / 0.1371, 4.98 GFLOPs, diversity 0.77.**
