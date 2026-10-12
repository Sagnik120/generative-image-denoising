# a10 Residual flow, few-step (generative)

**How it works:** InDI/rectified-flow style: x_t = (1−t)·clean + t·corrupted; one time-conditioned net predicts clean from x_t and the corrupted image. 2 inference passes (both counted in FLOPs), width 20. Half of the training batches use the net's own first prediction as x_t (matches inference). Only truly 'generative' candidate that works from the corrupted image instead of noise.
**Expectation:** better DISTS, lower PSNR.

## Results
Round-2 fixed benchmark (396 hold-out images: 198 natural, 99 low-light, 99 medical; mean input PSNR 24.49 dB). 60k steps, single seed.

| PSNR | SSIM | DISTS | GFLOPs | Params (M) | Avg rank (of 7) |
|---|---|---|---|---|---|
| 30.349 | 0.8238 | 0.1976 | 5.039 | 2.86 | 7.0 (PSNR 7, SSIM 7, DISTS 7) |

| Group | PSNR | SSIM | DISTS |
|---|---|---|---|
| single | 32.92 | 0.8866 | 0.1568 |
| extra | 32.38 | 0.8933 | 0.1390 |
| pair | 29.31 | 0.8002 | 0.2108 |
| triple | 28.51 | 0.7598 | 0.2431 |
| unseen | 28.45 | 0.7814 | 0.2371 |
| natural | 27.06 | 0.7785 | 0.1998 |
| lowlight | 35.36 | 0.8625 | 0.2233 |
| medical | 31.92 | 0.8757 | 0.1677 |

PSNR gap to the best other single-pass model, per case: worst `small_artifacts` -6.82, `salt_pepper` -6.22, `chromatic_aberration` -3.90; best `pixelate+uniform_noise` -0.05, `pixelate` -0.14, `jpeg` -0.55.

**Verdict: clearly worst (−1.5 dB, DISTS 0.198 vs 0.154); drop as a submission.** Two passes means each pass gets half the FLOPs, and the expected DISTS advantage did not appear because the regression models already train with a DISTS loss. Keeps value only as the 'generative' discussion point in the report.
