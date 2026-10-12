# a15 NAF + degradation pipeline v2

**How it works:** the a05 network, unchanged (4.98 GFLOPs, 5.52 M). Only the training data changes ("v2" pipeline in `src/common/degradations.py`):
- five more training families: `pixelate`, `periodic_noise` (both were *unseen* in v1), `sensor_noise`, `ringing`, `double_jpeg`;
- clean (zero-degradation) inputs raised from 3% to 8%; chain length probabilities [0.08, 0.34, 0.36, 0.17, 0.05] for 0–4 operations;
- severity drawn from Beta(1.0, 1.8), i.e. skewed towards mild corruptions;
- 50% of chains applied in random order (was 20%), BSRGAN-style;
- the nine listed families keep 65% of the sampling weight.
**Why it was tried:** the brief says the hidden set "may contain other types of noise/corruption". In round 2 the unseen group was 2.7 dB below the mean, and `resize_blur` output was worse than its input (over-processing of mild inputs). Wider and milder training data targets both.
**How it is measured:** since two v1-unseen families are now trained on, a second benchmark (v2, 96 images) was added: `unseen_v2` = film grain, glass blur, line dropout and two mixes (never trained on by anyone); `plan` = the three new training families.

## Results
Fixed benchmark v1 (396 hold-out images: 198 natural, 99 low-light, 99 medical; 33 cases; mean input PSNR 24.49 dB). 60k steps, single seed.

| PSNR | SSIM | DISTS | GFLOPs | Params (M) | Rank of 15 (PSNR / SSIM / DISTS) |
|---|---|---|---|---|---|
| 32.034 | 0.8594 | 0.1537 | 4.978 | 5.52 | 2 / 2 / 4 |

| Group | PSNR | SSIM | DISTS | PSNR vs a05 |
|---|---|---|---|---|
| single | 34.89 | 0.9159 | 0.1141 | -0.16 |
| extra | 34.52 | 0.9124 | 0.1086 | -0.60 |
| pair | 30.53 | 0.8327 | 0.1736 | -0.15 |
| triple | 29.36 | 0.7839 | 0.2131 | -0.19 |
| unseen | 31.61 | 0.8676 | 0.1428 | +2.42 |
| natural | 28.57 | 0.8199 | 0.1574 | +0.10 |
| lowlight | 36.94 | 0.8904 | 0.1773 | +0.12 |
| medical | 34.05 | 0.9076 | 0.1227 | +0.11 |

Benchmark v2 (96 images; `unseen_v2` = film grain, glass blur, line dropout and mixes, never trained on; `plan` = sensor noise, ringing, double JPEG, trained on only by the v2 pipeline):

| Group | PSNR | SSIM | DISTS |
|---|---|---|---|
| unseen_v2 | 30.99 | 0.8235 | 0.1865 |
| plan | 36.21 | 0.9429 | 0.0823 |
| v2 overall | 32.95 | 0.8683 | 0.1474 |

| v2 case | input | output | gain |
|---|---|---|---|
| `film_grain` | 25.09 | 28.25 | +3.16 |
| `glass_blur` | 31.39 | 31.87 | +0.47 |
| `line_dropout` | 19.69 | 35.40 | +15.72 |
| `glass_blur+film_grain` | 23.74 | 26.48 | +2.74 |
| `line_dropout+jpeg` | 19.21 | 32.95 | +13.74 |
| `sensor_noise` | 24.25 | 34.97 | +10.72 |
| `ringing` | 29.01 | 37.93 | +8.92 |
| `double_jpeg` | 34.65 | 35.73 | +1.07 |

PSNR difference to a05_nafnet_v2 per v1 case: worst `small_artifacts` -1.54, `motion_blur` -0.73, `chromatic_aberration` -0.53, `gaussian_blur+salt_pepper` -0.40; best `periodic_noise` +6.12, `pixelate` +2.14, `pixelate+uniform_noise` +1.52, `jpeg` +0.16.

## Analysis
- **Benchmark v1 overall: +0.11 dB, +0.0044 SSIM, −0.0005 DISTS vs a05.** Best of all 60k-step models on PSNR and SSIM.
- **The gain comes entirely from the families that moved into training:** `periodic_noise` +6.12 dB (30.44 → 36.56), `pixelate` +2.14, `pixelate+uniform_noise` +1.52. On benchmark v2's `plan` group: 36.21 vs a05's 33.18 (+3.03 dB; `ringing` alone 29.4 → 37.9).
- **It costs ~0.15–0.2 dB on the old families.** single −0.16, pair −0.15, triple −0.19, extra −0.60 (`small_artifacts` −1.54, `motion_blur` −0.73, `chromatic_aberration` −0.53). Same network, more tasks, same 60k steps: capacity and training time are shared out. This is a training-time dilution, not a capacity ceiling: a19 (same config, 200k steps) wins every one of these cases back with interest.
- **Generalisation to truly new families barely moved:** `unseen_v2` 30.99 dB vs a05 30.85 (+0.14 dB; DISTS 0.1865 vs 0.1985). The model learns the families it is shown and little beyond them. This is the main lesson of round 3: **coverage is what generalises, not "robustness"**. Each family added to training is worth 2–8 dB on that family; the spill-over to unrelated families is ~0.1 dB.
- **The over-processing fix did not work at 60k:** `resize_blur` 33.91 dB is still below the input (34.24) despite more clean inputs and milder severities. It was fixed only by longer training (a19: 35.02).
- All three domains improve (natural +0.10, low-light +0.12, medical +0.11).

**Verdict: accepted; v2 became the training pipeline of the final model.** The rule (unseen improves, v1 drops <0.1 dB) passed easily. The model itself is superseded by a19, which is this exact configuration trained 3.3× longer.
