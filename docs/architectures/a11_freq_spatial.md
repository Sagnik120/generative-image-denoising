# a11 Frequency + spatial dual-domain

**How it works:** NAF at 256px; below that, DualDomainBlocks = NAF spatial path + FFT branch (rFFT → 1x1 convs on spectrum → iFFT) for global receptive field (blur = frequency multiplication; JPEG grids, stripes, periodic noise are compact in frequency). Width 24, enc [2,2,2,2], mid 2.
**Note:** FFT FLOPs may be under-counted by fvcore/ptflops – verify the central measure; AMP + FFT can be fp16-unstable.

## Results
Round-2 fixed benchmark (396 hold-out images: 198 natural, 99 low-light, 99 medical; mean input PSNR 24.49 dB). 60k steps, single seed.

| PSNR | SSIM | DISTS | GFLOPs | Params (M) | Avg rank (of 7) |
|---|---|---|---|---|---|
| 31.838 | 0.8552 | 0.1572 | 4.884 | 5.27 | 2.75 (PSNR 4, SSIM 2, DISTS 4) |

| Group | PSNR | SSIM | DISTS |
|---|---|---|---|
| single | 34.91 | 0.9183 | 0.1095 |
| extra | 34.63 | 0.9112 | 0.1083 |
| pair | 30.62 | 0.8371 | 0.1674 |
| triple | 29.48 | 0.7895 | 0.2027 |
| unseen | 29.46 | 0.8077 | 0.2138 |
| natural | 28.40 | 0.8155 | 0.1611 |
| lowlight | 36.68 | 0.8851 | 0.1819 |
| medical | 33.88 | 0.9047 | 0.1246 |

PSNR gap to the best other single-pass model, per case: worst `small_artifacts` -1.35, `salt_pepper` -1.25, `chromatic_aberration` -0.64; best `pixelate+uniform_noise` +0.01, `pixelate` -0.02, `gaussian_blur+gaussian_noise` -0.04.

**Verdict: 3rd overall; second-best on unseen.** Like a07 it wins `periodic_noise` (31.58 dB) via the global FFT branch, but the expected blur advantage did not appear (motion 31.54 vs a05 31.64, a09 31.87). Lowest measured GFLOPs, but fvcore probably does not count the FFT ops, so its real cost is higher and another counter (ptflops) may disagree. Trained under AMP; riskier than a07 for the same benefit. **Not carried forward; a07's attention gives the same gain more safely.**
