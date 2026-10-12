# a18 NAF, strong DISTS (0.2) + SSIM (0.3) from step 0

**How it works:** a17 with heavier perceptual weights: DISTS 0.1 → 0.2 and SSIM 0.2 → 0.3, both from step 0. Network and data as a05.
**Why it was tried:** the upper end of the loss sweep. PSNR, SSIM and DISTS are ranked separately, so if doubling the DISTS weight buys a clear DISTS gain and the larger SSIM weight holds SSIM, two axes improve for a small PSNR cost.

## Results
Fixed benchmark v1 (396 hold-out images: 198 natural, 99 low-light, 99 medical; 33 cases; mean input PSNR 24.49 dB). 60k steps, single seed.

| PSNR | SSIM | DISTS | GFLOPs | Params (M) | Rank of 15 (PSNR / SSIM / DISTS) |
|---|---|---|---|---|---|
| 31.566 | 0.8476 | 0.1502 | 4.978 | 5.52 | 11 / 11 / 2 |

| Group | PSNR | SSIM | DISTS | PSNR vs a05 |
|---|---|---|---|---|
| single | 34.76 | 0.9142 | 0.1054 | -0.29 |
| extra | 34.36 | 0.9076 | 0.1027 | -0.76 |
| pair | 30.40 | 0.8312 | 0.1570 | -0.28 |
| triple | 29.26 | 0.7843 | 0.1880 | -0.29 |
| unseen | 28.68 | 0.7813 | 0.2195 | -0.51 |
| natural | 28.16 | 0.8088 | 0.1541 | -0.30 |
| lowlight | 36.30 | 0.8737 | 0.1761 | -0.52 |
| medical | 33.63 | 0.8991 | 0.1167 | -0.31 |

Benchmark v2 (96 images; `unseen_v2` = film grain, glass blur, line dropout and mixes, never trained on; `plan` = sensor noise, ringing, double JPEG, trained on only by the v2 pipeline):

| Group | PSNR | SSIM | DISTS |
|---|---|---|---|
| unseen_v2 | 30.79 | 0.8119 | 0.1959 |
| plan | 33.05 | 0.9102 | 0.1086 |
| v2 overall | 31.64 | 0.8488 | 0.1632 |

| v2 case | input | output | gain |
|---|---|---|---|
| `film_grain` | 25.09 | 27.17 | +2.08 |
| `glass_blur` | 31.39 | 31.61 | +0.21 |
| `line_dropout` | 19.69 | 36.27 | +16.58 |
| `glass_blur+film_grain` | 23.74 | 26.16 | +2.42 |
| `line_dropout+jpeg` | 19.21 | 32.75 | +13.54 |
| `sensor_noise` | 24.25 | 34.83 | +10.58 |
| `ringing` | 29.01 | 29.13 | +0.12 |
| `double_jpeg` | 34.65 | 35.20 | +0.55 |

PSNR difference to a05_nafnet_v2 per v1 case: worst `periodic_noise` -1.78, `small_artifacts` -1.46, `motion_blur` -0.98, `defocus_blur` -0.65; best `salt_pepper` +0.49, `gaussian_noise+salt_pepper` +0.02, `jpeg` +0.00, `pixelate` -0.02.

## Analysis
- **DISTS 0.1502 (−0.0040 vs a05, best of all 60k models), but PSNR −0.36 dB and SSIM −0.0074.** Rule failed on both counts.
- **SSIM went *down* although its weight went up.** The DISTS term dominates the objective (final training loss 0.085 vs a17's 0.059 and ~0.050 for the late-start runs), and DISTS rewards matching texture statistics, which tolerates small spatial misalignment that SSIM and PSNR punish. Raising both weights together does not let SSIM "keep up"; the two terms pull in different directions.
- **The exchange rate is poor and getting worse:** a05 → a17 trades 0.09 dB for 0.0025 DISTS; a17 → a18 trades a further 0.27 dB for 0.0015 DISTS. Weight 0.1 is already at or past the useful point.
- **Where the DISTS gain sits:** the hard mixed cases (pair 0.1570 vs a05 0.1633, triple 0.1880 vs 0.1982), where there is no single right answer and plausible texture is the best one can do. On singles it is level with a17 (0.1054 vs 0.1057).
- **Worst generalisation of the width-24 models:** unseen 28.68 dB (−0.51 vs a05), `periodic_noise` 28.65 (−1.78, barely above the 27.97 input). A strong texture prior makes the model reluctant to remove structured patterns that look like texture.

**Verdict: rejected.** It would gain perhaps one rank on DISTS and lose on PSNR and SSIM. This is the perception–distortion trade-off (Blau & Michaeli, 2018) measured directly on our setup, and useful as evidence for the report.
