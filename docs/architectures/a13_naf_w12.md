# a13 NAF width 12 (size study, ~1.3 GFLOPs)

**How it works:** a05 with base width 12 (half of a05's 24). 1.32 GFLOPs, 1.40 M parameters. Everything else identical to a05 and a12.
**Why it was tried:** the small end of the size study. With a12 it gives three points (1.3 / 2.3 / 5.0 GFLOPs) on the quality-vs-FLOPs curve, enough to see where the knee is.

## Results
Fixed benchmark v1 (396 hold-out images: 198 natural, 99 low-light, 99 medical; 33 cases; mean input PSNR 24.49 dB). 60k steps, single seed.

| PSNR | SSIM | DISTS | GFLOPs | Params (M) | Rank of 15 (PSNR / SSIM / DISTS) |
|---|---|---|---|---|---|
| 30.871 | 0.8364 | 0.1856 | 1.318 | 1.4 | 14 / 14 / 14 |

| Group | PSNR | SSIM | DISTS | PSNR vs a05 |
|---|---|---|---|---|
| single | 33.57 | 0.8972 | 0.1414 | -1.47 |
| extra | 32.87 | 0.8978 | 0.1306 | -2.25 |
| pair | 29.83 | 0.8149 | 0.1980 | -0.85 |
| triple | 28.88 | 0.7722 | 0.2298 | -0.67 |
| unseen | 28.91 | 0.7986 | 0.2334 | -0.29 |
| natural | 27.44 | 0.7916 | 0.1897 | -1.03 |
| lowlight | 35.98 | 0.8750 | 0.2082 | -0.84 |
| medical | 32.62 | 0.8876 | 0.1546 | -1.32 |

Benchmark v2 (96 images; `unseen_v2` = film grain, glass blur, line dropout and mixes, never trained on; `plan` = sensor noise, ringing, double JPEG, trained on only by the v2 pipeline):

| Group | PSNR | SSIM | DISTS |
|---|---|---|---|
| unseen_v2 | 29.28 | 0.7917 | 0.2254 |
| plan | 32.97 | 0.9081 | 0.1205 |
| v2 overall | 30.66 | 0.8353 | 0.1861 |

| v2 case | input | output | gain |
|---|---|---|---|
| `film_grain` | 25.09 | 27.08 | +1.99 |
| `glass_blur` | 31.39 | 31.57 | +0.17 |
| `line_dropout` | 19.69 | 32.27 | +12.58 |
| `glass_blur+film_grain` | 23.74 | 25.94 | +2.20 |
| `line_dropout+jpeg` | 19.21 | 29.55 | +10.34 |
| `sensor_noise` | 24.25 | 34.13 | +9.88 |
| `ringing` | 29.01 | 29.65 | +0.64 |
| `double_jpeg` | 34.65 | 35.12 | +0.47 |

PSNR difference to a05_nafnet_v2 per v1 case: worst `small_artifacts` -5.38, `salt_pepper` -2.85, `chromatic_aberration` -2.52, `motion_blur` -2.16; best `pixelate` -0.02, `periodic_noise` -0.16, `pixelate+uniform_noise` -0.17, `jpeg` -0.22.

## Analysis
- **−1.05 dB PSNR, −0.0186 SSIM, +0.0314 DISTS vs a05** for a 74% FLOPs cut.
- **The curve is convex, and the knee is above 2.3 GFLOPs.** 5.0 → 2.3 GFLOPs costs 0.49 dB (0.18 dB per GFLOP removed); 2.3 → 1.3 costs a further 0.57 dB (0.59 dB per GFLOP). Each further saving is three times as expensive in quality. DISTS shows the same shape (+0.014 then +0.017).
- **Where it breaks:** `small_artifacts` −5.38 dB, `salt_pepper` −2.85, `chromatic_aberration` −2.52, `motion_blur` −2.16. These are the cases where the answer must be *constructed* (inpainting, deblurring), not averaged. On `motion_blur` it gains only +0.59 dB over the input.
- **One oddity:** `periodic_noise` is slightly better than a12 (30.28 vs 29.85) and the unseen group equals a12. Under-fitting acts as regularisation on unseen patterns, but the effect is tiny and bought with a full dB elsewhere.
- Domain view: natural −1.03 dB, low-light −0.84, medical −1.32. Medical loses most because its fine, low-contrast structure needs capacity.

**Verdict: dropped.** Too small. It would rank first on FLOPs here but last-but-one on all three quality axes; with quality counted three times, that is a net loss against any competitive field.
