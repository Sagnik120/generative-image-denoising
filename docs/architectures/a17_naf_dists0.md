# a17 NAF, DISTS loss from step 0

**How it works:** a05 network, v1 data. One change: `dists_start_frac` 0.5 → 0.0, so the DISTS loss term (weight 0.1, computed on 4 images per batch) is active for all 60k steps instead of the last 30k. Loss = Charbonnier + 0.05 FFT + 0.2 SSIM + 0.1 DISTS throughout.
**Why it was tried:** in a05, DISTS fell 0.193 → 0.173 in the first epoch after the term switched on, with only a 0.03 dB PSNR dip. DISTS is a full ranking axis, so if 30k steps of it gives that much, 60k might give more. Rule: accept if DISTS improves ≥0.005 and PSNR falls ≤0.05 dB.

## Results
Fixed benchmark v1 (396 hold-out images: 198 natural, 99 low-light, 99 medical; 33 cases; mean input PSNR 24.49 dB). 60k steps, single seed.

| PSNR | SSIM | DISTS | GFLOPs | Params (M) | Rank of 15 (PSNR / SSIM / DISTS) |
|---|---|---|---|---|---|
| 31.833 | 0.8530 | 0.1517 | 4.978 | 5.52 | 9 / 9 / 3 |

| Group | PSNR | SSIM | DISTS | PSNR vs a05 |
|---|---|---|---|---|
| single | 35.09 | 0.9185 | 0.1057 | +0.04 |
| extra | 34.85 | 0.9104 | 0.1022 | -0.28 |
| pair | 30.58 | 0.8352 | 0.1598 | -0.10 |
| triple | 29.46 | 0.7891 | 0.1942 | -0.09 |
| unseen | 28.98 | 0.7958 | 0.2151 | -0.21 |
| natural | 28.39 | 0.8144 | 0.1557 | -0.08 |
| lowlight | 36.61 | 0.8821 | 0.1773 | -0.21 |
| medical | 33.94 | 0.9011 | 0.1183 | +0.01 |

Benchmark v2 (96 images; `unseen_v2` = film grain, glass blur, line dropout and mixes, never trained on; `plan` = sensor noise, ringing, double JPEG, trained on only by the v2 pipeline):

| Group | PSNR | SSIM | DISTS |
|---|---|---|---|
| unseen_v2 | 30.91 | 0.8094 | 0.1951 |
| plan | 33.22 | 0.9134 | 0.1062 |
| v2 overall | 31.78 | 0.8484 | 0.1617 |

| v2 case | input | output | gain |
|---|---|---|---|
| `film_grain` | 25.09 | 26.84 | +1.75 |
| `glass_blur` | 31.39 | 31.66 | +0.27 |
| `line_dropout` | 19.69 | 36.58 | +16.89 |
| `glass_blur+film_grain` | 23.74 | 26.25 | +2.51 |
| `line_dropout+jpeg` | 19.21 | 33.23 | +14.02 |
| `sensor_noise` | 24.25 | 34.99 | +10.74 |
| `ringing` | 29.01 | 29.33 | +0.32 |
| `double_jpeg` | 34.65 | 35.33 | +0.67 |

PSNR difference to a05_nafnet_v2 per v1 case: worst `small_artifacts` -0.87, `periodic_noise` -0.82, `motion_blur` -0.37, `motion_blur+jpeg` -0.23; best `salt_pepper` +0.81, `resize_blur` +0.19, `chromatic_aberration` +0.15, `gaussian_noise+salt_pepper` +0.13.

## Analysis
- **DISTS 0.1517 vs 0.1542 (−0.0025); PSNR −0.09 dB; SSIM −0.0020.** Half the required DISTS gain at twice the allowed PSNR cost: rule failed.
- **Why so little:** the DISTS term is fast to satisfy. In a05/a15 it takes ~10 epochs after switch-on to go 0.194 → 0.154; a17 has 24 epochs and ends at 0.152. Both runs converge to nearly the same place, so the extra 30k steps of DISTS gradient bought 0.0025.
- **Why PSNR drops:** early in training the network has not yet learned the coarse reconstruction; DISTS (VGG feature statistics) pulls towards plausible texture before the pixels are right, and those gradients compete with Charbonnier. At epoch 12 a17 is at 31.45 dB, where the late-start a15 run (same network, v2 data) is at 31.76. It never fully closes that gap.
- **Per case it is a mix, not a uniform shift:** `salt_pepper` +0.81 dB, `resize_blur` +0.19 (34.22, level with the input), `chromatic_aberration` +0.15; but `small_artifacts` −0.87, `periodic_noise` −0.82, `motion_blur` −0.37. Unseen group −0.21 dB.
- **Best DISTS in every domain among 60k runs except a18** (medical 0.1183 vs a05 0.1233, natural 0.1557 vs 0.1569). The benefit is real, just small.
- Training is ~12% slower for the first half (735 vs 650 s/epoch), because the VGG forward/backward runs from the start.

**Verdict: rejected for the final model; the late-start schedule is kept.** Note the rule was tested at 60k steps. In the 200k final run DISTS was on for the last 100k steps and was still falling at the end (0.1302 → 0.1293 over the last 10 epochs), so an *intermediate* start (e.g. 25–30% of training) remains a reasonable, cheap variation for a longer run.
