# a06 Restormer v2 (transformer reference)

**How it works:** Restormer blocks (MDTA + GDFN) at every scale, width 20, enc [2,2,3], mid 3 at 32px, heads [1,2,4,8]. A block costs ~2× a NAFBlock, so narrower/shallower. Round 2 loss: Charbonnier 1.0 + 0.05 FFT + 0.2 SSIM + 0.1 DISTS (after 50% of training).
**Question:** attention everywhere vs convolution everywhere at equal FLOPs.

## Results
Round-2 fixed benchmark (396 hold-out images: 198 natural, 99 low-light, 99 medical; mean input PSNR 24.49 dB). 60k steps, single seed.

| PSNR | SSIM | DISTS | GFLOPs | Params (M) | Avg rank (of 7) |
|---|---|---|---|---|---|
| 31.682 | 0.852 | 0.1623 | 4.984 | 1.31 | 5.75 (PSNR 6, SSIM 6, DISTS 6) |

| Group | PSNR | SSIM | DISTS |
|---|---|---|---|
| single | 34.85 | 0.9170 | 0.1132 |
| extra | 34.17 | 0.9081 | 0.1131 |
| pair | 30.50 | 0.8344 | 0.1718 |
| triple | 29.49 | 0.7906 | 0.2052 |
| unseen | 28.98 | 0.7936 | 0.2279 |
| natural | 28.28 | 0.8122 | 0.1646 |
| lowlight | 36.51 | 0.8825 | 0.1877 |
| medical | 33.65 | 0.9012 | 0.1325 |

PSNR gap to the best other single-pass model, per case: worst `small_artifacts` -2.35, `periodic_noise` -1.88, `salt_pepper` -1.10; best `motion_blur+jpeg` -0.03, `motion_blur+poisson_noise+jpeg` -0.07, `defocus_blur+speckle_noise` -0.07.

**Verdict: worst single-pass model; drop.** Attention at every scale forces width 20 and few blocks at ~5 GFLOPs, and it loses on every group. Only 1.3M params, so it is under-capacity for 17 degradation families. Consistent with NAFNet's finding that conv blocks beat Restormer per FLOP.
