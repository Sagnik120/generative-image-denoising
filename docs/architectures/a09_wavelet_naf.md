# a09 Wavelet NAF

**How it works:** invertible Haar transform → 128x128x12; NAF U-Net (width 40, enc [4,5,7], mid 5, dec [3,2,2]) predicts wavelet-coefficient correction; inverse Haar. More depth for same FLOPs; separates smooth content from noise/edges. Concern: Haar can introduce block/aliasing artifacts and limits full-res detail.
**Opportunity:** could be shrunk to ~2.5 GFLOPs cheaply.

## Results
Round-2 fixed benchmark (396 hold-out images: 198 natural, 99 low-light, 99 medical; mean input PSNR 24.49 dB). 60k steps, single seed.

| PSNR | SSIM | DISTS | GFLOPs | Params (M) | Avg rank (of 7) |
|---|---|---|---|---|---|
| 31.891 | 0.8543 | 0.1545 | 5.018 | 6.44 | 3.5 (PSNR 2, SSIM 4, DISTS 2) |

| Group | PSNR | SSIM | DISTS |
|---|---|---|---|
| single | 34.98 | 0.9182 | 0.1070 |
| extra | 35.17 | 0.9152 | 0.1044 |
| pair | 30.64 | 0.8380 | 0.1631 |
| triple | 29.58 | 0.7922 | 0.1956 |
| unseen | 29.10 | 0.7912 | 0.2217 |
| natural | 28.40 | 0.8153 | 0.1591 |
| lowlight | 36.76 | 0.8823 | 0.1796 |
| medical | 34.01 | 0.9043 | 0.1202 |

PSNR gap to the best other single-pass model, per case: worst `salt_pepper` -2.16, `periodic_noise` -1.76, `gaussian_blur+salt_pepper` -0.55; best `resize_blur` +0.57, `chromatic_aberration` +0.37, `defocus_blur` +0.24.

**Verdict: equal to a05 in quality (−0.03 dB, same DISTS), best on blur and medical, ~20% faster to train.** Best on motion/defocus blur, chromatic aberration, `resize_blur` (34.62, the only model clearly above the input) and most pair/triple cases, thanks to the extra depth. Worst single-pass model on `salt_pepper` (35.85 vs 37.17): Haar spreads an impulse over all four sub-bands. Proves half-resolution processing loses nothing, so it is the **best candidate for a low-FLOPs variant.**
