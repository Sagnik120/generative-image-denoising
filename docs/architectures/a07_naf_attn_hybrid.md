# a07 NAF + attention hybrid

**How it works:** widths [24,48,96,192,256]. NAFBlocks at 256/128/64px, channel-attention blocks at 32px, true global spatial self-attention at 16px bottleneck (256 tokens, cheap). Safe because input size is fixed at 256. Round 2 loss: Charbonnier 1.0 + 0.05 FFT + 0.2 SSIM + 0.1 DISTS (after 50% of training).
**Question:** does global context help large blur kernels / structure?

## Results
Round-2 fixed benchmark (396 hold-out images: 198 natural, 99 low-light, 99 medical; mean input PSNR 24.49 dB). 60k steps, single seed.

| PSNR | SSIM | DISTS | GFLOPs | Params (M) | Avg rank (of 7) |
|---|---|---|---|---|---|
| 31.854 | 0.8558 | 0.1563 | 4.928 | 4.02 | 2.25 (PSNR 3, SSIM 1, DISTS 3) |

| Group | PSNR | SSIM | DISTS |
|---|---|---|---|
| single | 34.84 | 0.9176 | 0.1094 |
| extra | 34.74 | 0.9122 | 0.1069 |
| pair | 30.63 | 0.8366 | 0.1661 |
| triple | 29.55 | 0.7922 | 0.2001 |
| unseen | 29.52 | 0.8117 | 0.2144 |
| natural | 28.42 | 0.8162 | 0.1598 |
| lowlight | 36.74 | 0.8872 | 0.1808 |
| medical | 33.84 | 0.9036 | 0.1247 |

PSNR gap to the best other single-pass model, per case: worst `salt_pepper` -1.54, `small_artifacts` -0.88, `resize_blur` -0.76; best `periodic_noise` +0.25, `jpeg` -0.01, `pixelate` -0.03.

**Verdict: 2nd overall; best SSIM and best on unseen degradations.** Overall 0.07 dB behind a05, but +0.33 dB on the unseen group, almost entirely from `periodic_noise` (31.83 vs 30.44 dB): the global 16px attention can see a pattern spanning the whole image. Slightly worse on blur singles and `salt_pepper`. **Keep the 16px global-attention bottleneck as an add-on to the a05 body.**
