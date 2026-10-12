# a14 Wavelet NAF width 28 (size study, ~2.5 GFLOPs)

**How it works:** a09 shrunk. Invertible Haar transform (256×256×3 → 128×128×12), NAF U-Net at half resolution (width 28, enc [4,5,7], mid 5, dec [3,2,2]), inverse Haar. Because every block runs at ≤128 px, the network affords 28 blocks against a12's 18 at similar cost. 2.52 GFLOPs, 3.18 M parameters.
**Why it was tried:** in round 2, a09 tied a05 at 5 GFLOPs and was best on blur and medical, so the hypothesis was that half-resolution processing is the *better* way to be small: keep depth, drop resolution, instead of dropping width.

## Results
Fixed benchmark v1 (396 hold-out images: 198 natural, 99 low-light, 99 medical; 33 cases; mean input PSNR 24.49 dB). 60k steps, single seed.

| PSNR | SSIM | DISTS | GFLOPs | Params (M) | Rank of 15 (PSNR / SSIM / DISTS) |
|---|---|---|---|---|---|
| 31.359 | 0.8455 | 0.1687 | 2.522 | 3.18 | 13 / 13 / 13 |

| Group | PSNR | SSIM | DISTS | PSNR vs a05 |
|---|---|---|---|---|
| single | 34.30 | 0.9086 | 0.1220 | -0.75 |
| extra | 33.85 | 0.9076 | 0.1169 | -1.27 |
| pair | 30.22 | 0.8272 | 0.1789 | -0.46 |
| triple | 29.27 | 0.7852 | 0.2103 | -0.28 |
| unseen | 28.89 | 0.7874 | 0.2302 | -0.30 |
| natural | 27.89 | 0.8042 | 0.1732 | -0.58 |
| lowlight | 36.37 | 0.8764 | 0.1926 | -0.45 |
| medical | 33.28 | 0.8972 | 0.1359 | -0.66 |

Benchmark v2 (96 images; `unseen_v2` = film grain, glass blur, line dropout and mixes, never trained on; `plan` = sensor noise, ringing, double JPEG, trained on only by the v2 pipeline):

| Group | PSNR | SSIM | DISTS |
|---|---|---|---|
| unseen_v2 | 30.04 | 0.8054 | 0.2074 |
| plan | 33.06 | 0.9100 | 0.1148 |
| v2 overall | 31.18 | 0.8446 | 0.1726 |

| v2 case | input | output | gain |
|---|---|---|---|
| `film_grain` | 25.09 | 27.37 | +2.27 |
| `glass_blur` | 31.39 | 31.58 | +0.19 |
| `line_dropout` | 19.69 | 33.26 | +13.58 |
| `glass_blur+film_grain` | 23.74 | 26.16 | +2.42 |
| `line_dropout+jpeg` | 19.21 | 31.85 | +12.64 |
| `sensor_noise` | 24.25 | 34.65 | +10.40 |
| `ringing` | 29.01 | 29.48 | +0.47 |
| `double_jpeg` | 34.65 | 35.06 | +0.41 |

PSNR difference to a05_nafnet_v2 per v1 case: worst `small_artifacts` -3.45, `salt_pepper` -2.72, `gaussian_blur+salt_pepper` -1.38, `chromatic_aberration` -0.89; best `pixelate+uniform_noise` +0.02, `gaussian_blur+gaussian_noise+jpeg` -0.06, `resize_blur` -0.09, `pixelate` -0.10.

## Analysis
- **The hypothesis is false.** a14 scores 31.36 dB / 0.8455 / 0.1687 at 2.52 GFLOPs; a12 scores 31.44 / 0.8468 / 0.1682 at 2.28. a14 is worse on all three quality metrics *and* costs 11% more FLOPs, so it is dominated.
- **Why depth did not help here:** at 5 GFLOPs a09 had width 40, enough channels per level; at width 28 the 12-channel Haar input is squeezed into a narrow first stage, and the full-resolution detail has to be carried entirely by the wavelet coefficients. The benefit of a09 was depth *on top of* adequate width, not depth instead of it.
- **The a09 fingerprint is unchanged:** relatively good on `resize_blur` (33.94 vs a12 33.62), `motion_blur` (31.01 vs 30.87) and pair/triple cases (equal to a12), and bad on impulses: `salt_pepper` 34.44 vs a12 36.50 (−2.05 dB). Haar spreads a single-pixel impulse into all four sub-bands, turning an easy outlier into a 2×2 pattern.
- Benchmark v2: `unseen_v2` 30.04 vs a12 30.39. No generalisation advantage either.

**Verdict: dropped. Wavelet processing is not a cheaper route to quality; width scaling (a12) is the better small model.** This closes the a09 line: equal to NAF at 5 GFLOPs, worse at 2.5.
