# a16 NAF + global attention + degradation v2

**How it works:** a15 with `global_attn: true` in PlanNet: the three bottleneck blocks (16×16) become transformer blocks with full spatial self-attention (8 heads, 256 tokens), and the bottleneck is narrowed to 2/3 width to keep FLOPs level. 4.85 GFLOPs (−2.5%), 4.19 M parameters (−24%). Trained on the v2 pipeline like a15.
**Why it was tried:** in round 2, a07's attention bottleneck led the unseen group by 0.3 dB, all of it from `periodic_noise` (31.8 vs 30.4 dB): a pattern that spans the image should need a global operator. This is the controlled test of that claim against a15. Rule: keep the attention if it improves unseen by ≥0.15 dB at <3% extra FLOPs and costs ≤0.05 dB overall.

## Results
Fixed benchmark v1 (396 hold-out images: 198 natural, 99 low-light, 99 medical; 33 cases; mean input PSNR 24.49 dB). 60k steps, single seed.

| PSNR | SSIM | DISTS | GFLOPs | Params (M) | Rank of 15 (PSNR / SSIM / DISTS) |
|---|---|---|---|---|---|
| 31.989 | 0.8582 | 0.1550 | 4.853 | 4.19 | 3 / 3 / 7 |

| Group | PSNR | SSIM | DISTS | PSNR vs a15 |
|---|---|---|---|---|
| single | 34.84 | 0.9155 | 0.1154 | -0.05 |
| extra | 34.49 | 0.9097 | 0.1109 | -0.03 |
| pair | 30.46 | 0.8306 | 0.1745 | -0.06 |
| triple | 29.30 | 0.7830 | 0.2134 | -0.06 |
| unseen | 31.64 | 0.8676 | 0.1459 | +0.03 |
| natural | 28.54 | 0.8186 | 0.1587 | -0.03 |
| lowlight | 36.89 | 0.8899 | 0.1776 | -0.05 |
| medical | 33.99 | 0.9058 | 0.1251 | -0.06 |

Benchmark v2 (96 images; `unseen_v2` = film grain, glass blur, line dropout and mixes, never trained on; `plan` = sensor noise, ringing, double JPEG, trained on only by the v2 pipeline):

| Group | PSNR | SSIM | DISTS |
|---|---|---|---|
| unseen_v2 | 30.18 | 0.8119 | 0.1989 |
| plan | 36.43 | 0.9433 | 0.0826 |
| v2 overall | 32.52 | 0.8612 | 0.1553 |

| v2 case | input | output | gain |
|---|---|---|---|
| `film_grain` | 25.09 | 28.21 | +3.12 |
| `glass_blur` | 31.39 | 31.73 | +0.33 |
| `line_dropout` | 19.69 | 33.22 | +13.53 |
| `glass_blur+film_grain` | 23.74 | 26.36 | +2.62 |
| `line_dropout+jpeg` | 19.21 | 31.37 | +12.16 |
| `sensor_noise` | 24.25 | 34.91 | +10.65 |
| `ringing` | 29.01 | 38.70 | +9.69 |
| `double_jpeg` | 34.65 | 35.68 | +1.03 |

PSNR difference to a15_naf_degv2 per v1 case: worst `gaussian_blur+salt_pepper` -0.20, `gaussian_blur` -0.17, `chromatic_aberration` -0.17, `motion_blur+poisson_noise+jpeg` -0.16; best `salt_pepper` +0.20, `chromatic_aberration+gaussian_noise` +0.13, `periodic_noise` +0.12, `gaussian_blur+gaussian_noise+jpeg` +0.09.

## Analysis
- **Benchmark v1: −0.045 dB, −0.0012 SSIM, +0.0013 DISTS vs a15.** Slightly worse on all three.
- **The round-2 advantage disappeared.** `periodic_noise` 36.68 vs 36.56 (+0.12 dB), v1 unseen group +0.03 dB. Once periodic patterns are in the training set, the plain conv network removes them just as well; a07's lead was a generalisation accident for one unseen family, not a structural need for global context. The receptive field of a 4-level NAF U-Net already covers the 256 px image.
- **It hurts on truly unseen data:** `unseen_v2` 30.18 vs 30.99 dB (−0.82), driven by `line_dropout` −2.18 dB and `line_dropout+jpeg` −1.58. Dead rows are one pixel wide; they must be filled from the neighbouring rows at full resolution, and the narrowed bottleneck gave up conv capacity for attention at 16 px, where a row no longer exists.
- **Small wins:** `ringing` +0.77 dB, `plan` group +0.22, `salt_pepper` +0.20, `chromatic_aberration+gaussian_noise` +0.13. Ringing is oscillation around every edge, a quasi-global pattern, so the mechanism is real but narrow.
- The 24% parameter saving is irrelevant: parameters are not scored, FLOPs are, and the FLOPs saving is 2.5%.

**Verdict: rejected (gain +0.03 dB against a 0.15 dB threshold; −0.82 dB on `unseen_v2`).** The final model has no attention. The repo's default `a19_final/config.yaml` still shows `global_attn: true`; `run_plan.py` overwrote it with `false` for the actual run (see `results/a19_final/final_report.json`).
