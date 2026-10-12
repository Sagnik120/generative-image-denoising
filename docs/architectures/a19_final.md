# a19 Final model (a15 configuration, 200k steps)

> **Update 2026-10-11:** the brief requires a *generative* model. a19 is a regression network and is **not a compliant submission**; it is now the upper reference and the initialisation for the generative model. See `../CURRENT_STATUS.md` and `../PLAN.md`. The measurements below remain valid.

**How it works:** `PlanNet` = plain NAF U-Net with residual output. Width 24; encoder [2,2,2,4] NAF blocks at 256/128/64/32 px; 3 blocks at the 16 px bottleneck; decoder [1,1,1,2]; channel widths 24-48-96-192-384. **4.98 GFLOPs, 5.52 M parameters, one forward pass, no attention, no wavelet.**
**What was actually trained** (from `results/a19_final/final_report.json`; `run_plan.py` wrote this config from the round-3 decision rules, and it differs from the default `config.yaml` in the repo):

| Choice | Value | Decided by |
|---|---|---|
| Size | width 24 (5 GFLOPs) | a12/a13/a14 all outside the 0.15 dB tolerance |
| Degradations | v2 pipeline (22 training families) | a15 passed |
| Global attention | **off** | a16 failed (+0.03 dB unseen, −0.82 dB `unseen_v2`) |
| DISTS schedule | weight 0.1, **on from 50%** of training (step 100k) | a17/a18 failed |
| Other losses | Charbonnier 1.0 + FFT 0.05 + SSIM 0.2 | unchanged |
| Schedule | **200k steps** (80 × 2500), batch 16, AdamW 4e-4 → 1e-6 cosine, 1k warm-up, EMA 0.999, AMP, grad-clip 1.0 | 3.3× the 60k runs |

So **a19 is a15 trained 3.3× longer, and nothing else.** Same seed: the epoch-1 validation numbers match a15 to four digits. Training took 17.4 h.

## Results
Fixed benchmark v1 (396 hold-out images: 198 natural, 99 low-light, 99 medical; 33 cases; mean input PSNR 24.49 dB). 200k steps, single seed.

| PSNR | SSIM | DISTS | GFLOPs | Params (M) | Rank of 15 (PSNR / SSIM / DISTS) |
|---|---|---|---|---|---|
| 33.112 | 0.8743 | 0.1293 | 4.978 | 5.52 | 1 / 1 / 1 |

| Group | PSNR | SSIM | DISTS | PSNR vs a15 |
|---|---|---|---|---|
| single | 36.28 | 0.9304 | 0.0891 | +1.39 |
| extra | 37.09 | 0.9220 | 0.0889 | +2.56 |
| pair | 31.18 | 0.8491 | 0.1494 | +0.65 |
| triple | 29.96 | 0.8005 | 0.1850 | +0.61 |
| unseen | 32.75 | 0.8798 | 0.1204 | +1.14 |
| natural | 29.50 | 0.8375 | 0.1338 | +0.93 |
| lowlight | 37.99 | 0.9026 | 0.1519 | +1.05 |
| medical | 35.45 | 0.9194 | 0.0979 | +1.40 |

Benchmark v2 (96 images; `unseen_v2` = film grain, glass blur, line dropout and mixes, never trained on; `plan` = sensor noise, ringing, double JPEG, trained on only by the v2 pipeline):

| Group | PSNR | SSIM | DISTS |
|---|---|---|---|
| unseen_v2 | 31.88 | 0.8290 | 0.1736 |
| plan | 37.38 | 0.9487 | 0.0708 |
| v2 overall | 33.94 | 0.8739 | 0.1351 |

| v2 case | input | output | gain |
|---|---|---|---|
| `film_grain` | 25.09 | 28.44 | +3.35 |
| `glass_blur` | 31.39 | 31.48 | +0.09 |
| `line_dropout` | 19.69 | 38.97 | +19.28 |
| `glass_blur+film_grain` | 23.74 | 26.46 | +2.72 |
| `line_dropout+jpeg` | 19.21 | 34.04 | +14.83 |
| `sensor_noise` | 24.25 | 35.46 | +11.21 |
| `ringing` | 29.01 | 40.53 | +11.53 |
| `double_jpeg` | 34.65 | 36.13 | +1.48 |

PSNR difference to a15_naf_degv2 per v1 case: worst `defocus_blur+poisson_noise` +0.25, `gaussian_blur+gaussian_noise` +0.32, `jpeg` +0.33, `defocus_blur+speckle_noise` +0.34; best `small_artifacts` +6.02, `salt_pepper` +4.40, `periodic_noise` +3.35, `chromatic_aberration` +1.93.

## Training curve (validation = benchmark v1)

| Epoch (×2.5k steps) | PSNR | SSIM | DISTS | Note |
|---|---|---|---|---|
| 11 | 31.62 | 0.8522 | 0.1952 | |
| 21 | 32.38 | 0.8655 | 0.1809 | already above a15's final 32.03 |
| 36 | 32.92 | 0.8728 | 0.1719 | |
| 41 | 32.87 | 0.8722 | 0.1512 | first epoch with the DISTS loss: DISTS −0.02, PSNR −0.05 dB vs epoch 36 |
| 51 | 32.97 | 0.8725 | 0.1361 | |
| 61 | 33.06 | 0.8735 | 0.1317 | |
| 71 | 33.11 | 0.8742 | 0.1302 | |
| 80 | 33.11 | 0.8743 | 0.1293 | last 9 epochs: +0.006 dB, −0.0008 DISTS |

## Analysis
- **+1.08 dB PSNR, +0.0149 SSIM, −0.0244 DISTS over a15 at identical FLOPs.** This single change is worth 10× any architecture difference found in rounds 2–3 (top-five spread there: 0.09 dB). Best PSNR in **every one of the 33 v1 cases**, and best PSNR, SSIM and DISTS in every group and every domain.
- **The round-2 conclusion "training has converged" was wrong.** The 60k curves were flat at the end because the cosine schedule had driven the learning rate to zero, not because the network was saturated. a19's curve is flat at the end for the same reason, so the same caution applies: there is probably more to take from a longer or restarted schedule, with diminishing returns.
- **Where the extra training went** (a19 − a15): `small_artifacts` +6.0 dB, `salt_pepper` +4.4, `periodic_noise` +3.4, `chromatic_aberration` +1.9, `motion_blur` +1.7, `defocus_blur` +1.4, `gaussian_blur` +1.2, `resize_blur` +1.1. These are *deterministic, invertible-in-principle* corruptions: the information is still in the image and more training means a more exact inverse. Random-noise cases gain only +0.3–0.6 dB (`gaussian_noise` +0.43, `poisson_noise` +0.55) and the mixed cases +0.3–0.9: there the information is destroyed and the model is near the achievable floor.
- **The over-processing bug is fixed:** `resize_blur` 35.02 dB vs input 34.24 (a05 and a15 were both below the input).
- **True generalisation (benchmark v2, `unseen_v2`, never trained on): 31.88 dB / 0.8290 / 0.1736** vs a15 30.99 and a05 30.85. Most of it is `line_dropout` (38.97 vs 35.40) and `line_dropout+jpeg` (34.04 vs 32.95); `film_grain` +0.19.
- **Remaining weaknesses:**
  - `glass_blur` (unseen): 31.48 dB vs input 31.40, **+0.09 dB, and 0.39 dB worse than a15**. Longer training specialised the model and it lost the little it did here. Local pixel shuffling is simply not in the training set.
  - `film_grain` (unseen): 28.44 dB, DISTS 0.22; `glass_blur+film_grain` 26.46, DISTS 0.26.
  - `jpeg`: +1.29 dB over the input only (36.43 vs 35.14). A listed family.
  - `pixelate`: 29.64 (+2.8 dB); information is gone, limited headroom.
  - Blur + noise mixes: `motion_blur+gaussian_noise` 26.97, `defocus_blur+speckle_noise` 28.27, `motion_blur+poisson_noise+jpeg` 28.67. Triple group gained least (+0.61 dB).
  - Natural images remain the hard domain: 29.50 dB vs 37.99 low-light and 35.45 medical. Low-light has the worst DISTS (0.152).
- **Model selection note:** the best checkpoint is chosen on the same benchmark that is reported. With a monotone curve the bias is negligible, but the number is not from a fully untouched set; benchmark v2 is closer to that.

**Verdict: best model, but regression, so not a compliant submission.** It is the upper reference and the initialisation for the generative model. A deliberately simple network; its strength is the data pipeline and the training length, not the architecture. Next steps: `../PLAN.md`.
