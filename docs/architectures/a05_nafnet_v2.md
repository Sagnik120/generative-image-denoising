# a05 NAFNet v2 (conv reference)

**How it works:** NAFBlock U-Net, 5 levels down to 16px, width 24, enc [2,2,2,4], mid 3, dec [1,1,1,2]; blocks concentrated at coarse levels. Residual prediction. Round 2 loss: Charbonnier 1.0 + 0.05 FFT + 0.2 SSIM + 0.1 DISTS (after 50% of training).
**Role:** baseline; every round-2 model compares to it at equal FLOPs.

## Results
Round-2 fixed benchmark (396 hold-out images: 198 natural, 99 low-light, 99 medical; mean input PSNR 24.49 dB). 60k steps, single seed.

| PSNR | SSIM | DISTS | GFLOPs | Params (M) | Avg rank (of 7) |
|---|---|---|---|---|---|
| 31.924 | 0.855 | 0.1542 | 4.978 | 5.52 | 2.0 (PSNR 1, SSIM 3, DISTS 1) |

| Group | PSNR | SSIM | DISTS |
|---|---|---|---|
| single | 35.05 | 0.9196 | 0.1063 |
| extra | 35.12 | 0.9134 | 0.1045 |
| pair | 30.68 | 0.8385 | 0.1633 |
| triple | 29.55 | 0.7927 | 0.1982 |
| unseen | 29.19 | 0.7935 | 0.2169 |
| natural | 28.47 | 0.8165 | 0.1569 |
| lowlight | 36.82 | 0.8863 | 0.1797 |
| medical | 33.94 | 0.9009 | 0.1233 |

PSNR gap to the best other single-pass model, per case: worst `periodic_noise` -1.40, `salt_pepper` -0.83, `resize_blur` -0.59; best `small_artifacts` +0.40, `gaussian_blur+jpeg` +0.11, `speckle_noise` +0.10.

**Verdict: best overall (avg rank 2.0), and the simplest model.** Best PSNR and DISTS; best on natural and low-light images. Weak spots: `resize_blur` comes out *below* the input (34.02 vs 34.24 dB, i.e. it damages mildly degraded images), `pixelate` barely improves (+0.26 dB), unseen `periodic_noise` is 1.4 dB behind a07. **Carry forward as the base model.**
