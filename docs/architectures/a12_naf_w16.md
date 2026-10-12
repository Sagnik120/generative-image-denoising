# a12 NAF width 16 (size study, ~2.3 GFLOPs)

**How it works:** exactly a05 (plain NAF U-Net, enc [2,2,2,4], mid 3, dec [1,1,1,2], residual output) with the base width cut from 24 to 16. Convolution cost scales with width², so FLOPs fall to (16/24)² ≈ 0.44 of a05: 2.28 GFLOPs, 2.47 M parameters. Same data, v1 degradations, losses and 60k-step schedule as a05, so the only variable is size.
**Why it was tried:** FLOPs is one of the four ranked axes and round 2 never moved off ~5 GFLOPs. This run measures the price of halving it. Decision rule set in advance: accept the smaller size only if it stays within 0.15 dB PSNR, 0.004 SSIM and 0.005 DISTS of a05.

## Results
Fixed benchmark v1 (396 hold-out images: 198 natural, 99 low-light, 99 medical; 33 cases; mean input PSNR 24.49 dB). 60k steps, single seed.

| PSNR | SSIM | DISTS | GFLOPs | Params (M) | Rank of 15 (PSNR / SSIM / DISTS) |
|---|---|---|---|---|---|
| 31.437 | 0.8468 | 0.1682 | 2.278 | 2.47 | 12 / 12 / 12 |

| Group | PSNR | SSIM | DISTS | PSNR vs a05 |
|---|---|---|---|---|
| single | 34.50 | 0.9114 | 0.1205 | -0.54 |
| extra | 33.94 | 0.9063 | 0.1166 | -1.18 |
| pair | 30.25 | 0.8277 | 0.1793 | -0.43 |
| triple | 29.26 | 0.7843 | 0.2132 | -0.29 |
| unseen | 28.94 | 0.7923 | 0.2248 | -0.26 |
| natural | 27.99 | 0.8052 | 0.1725 | -0.48 |
| lowlight | 36.37 | 0.8787 | 0.1923 | -0.45 |
| medical | 33.40 | 0.8981 | 0.1354 | -0.54 |

Benchmark v2 (96 images; `unseen_v2` = film grain, glass blur, line dropout and mixes, never trained on; `plan` = sensor noise, ringing, double JPEG, trained on only by the v2 pipeline):

| Group | PSNR | SSIM | DISTS |
|---|---|---|---|
| unseen_v2 | 30.39 | 0.8143 | 0.2039 |
| plan | 33.02 | 0.9098 | 0.1149 |
| v2 overall | 31.38 | 0.8501 | 0.1705 |

| v2 case | input | output | gain |
|---|---|---|---|
| `film_grain` | 25.09 | 26.85 | +1.76 |
| `glass_blur` | 31.39 | 31.63 | +0.23 |
| `line_dropout` | 19.69 | 34.87 | +15.18 |
| `glass_blur+film_grain` | 23.74 | 26.20 | +2.46 |
| `line_dropout+jpeg` | 19.21 | 32.40 | +13.19 |
| `sensor_noise` | 24.25 | 34.61 | +10.36 |
| `ringing` | 29.01 | 29.39 | +0.38 |
| `double_jpeg` | 34.65 | 35.06 | +0.40 |

PSNR difference to a05_nafnet_v2 per v1 case: worst `small_artifacts` -2.79, `gaussian_blur+salt_pepper` -0.92, `chromatic_aberration` -0.87, `motion_blur` -0.77; best `pixelate+uniform_noise` -0.02, `pixelate` -0.04, `jpeg` -0.05, `gaussian_blur+gaussian_noise+jpeg` -0.08.

## Analysis
- **Cost of halving FLOPs: −0.49 dB PSNR, −0.0082 SSIM, +0.0140 DISTS.** All three are 2–3× outside the tolerance, so the rule rejected it.
- **The loss is not uniform.** Cases that need precise spatial reconstruction lose most: `small_artifacts` −2.79 dB, `chromatic_aberration` −0.87, `motion_blur` −0.77, `gaussian_blur` −0.75. Plain noise cases lose only 0.3–0.4 dB and JPEG −0.05. Denoising is mostly local averaging, which a narrow network still does; inverting a blur or re-registering colour planes needs more channels to hold the detail being rebuilt.
- **Generalisation is hurt least:** unseen −0.26 dB, and on benchmark v2 `unseen_v2` −0.46 dB. A smaller network is not noticeably less robust, only less accurate.
- **Caveat:** this is a 60k-step comparison. a19 showed that 200k steps is worth +1.08 dB at width 24; a width-16 model would also gain from long training, so the *gap* at 200k is unknown (probably similar, ~0.4–0.5 dB, since under-capacity does not train away).

**Verdict: rejected by the size rule, but it is the only sensible low-FLOPs candidate.** 54% fewer FLOPs for 0.49 dB. Three of the four axes are quality, so the trade only pays if many rival teams sit between 2.3 and 5 GFLOPs *and* few sit within 0.5 dB of us. Kept as the fallback "efficient" submission (see `../PLAN.md`).
