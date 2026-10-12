# a08 Degradation-aware hybrid

**How it works:** a07 body + tiny encoder producing a degradation embedding (dim 64) that FiLM-modulates features in every stage. Training-only auxiliary head predicts which degradation families/severity were applied (weight 0.1); no inference FLOPs, encoder ≈ +0.05 GFLOPs.
**Question:** does explicit degradation awareness help mixed/unseen corruptions? Compare directly with a07.

## Results
Round-2 fixed benchmark (396 hold-out images: 198 natural, 99 low-light, 99 medical; mean input PSNR 24.49 dB). 60k steps, single seed.

| PSNR | SSIM | DISTS | GFLOPs | Params (M) | Avg rank (of 7) |
|---|---|---|---|---|---|
| 31.835 | 0.853 | 0.1612 | 4.982 | 4.23 | 4.75 (PSNR 5, SSIM 5, DISTS 5) |

| Group | PSNR | SSIM | DISTS |
|---|---|---|---|
| single | 34.98 | 0.9187 | 0.1108 |
| extra | 34.95 | 0.9112 | 0.1105 |
| pair | 30.60 | 0.8364 | 0.1704 |
| triple | 29.48 | 0.7892 | 0.2088 |
| unseen | 29.08 | 0.7915 | 0.2256 |
| natural | 28.43 | 0.8145 | 0.1642 |
| lowlight | 36.71 | 0.8812 | 0.1868 |
| medical | 33.76 | 0.9021 | 0.1296 |

PSNR gap to the best other single-pass model, per case: worst `periodic_noise` -1.84, `chromatic_aberration` -1.11, `resize_blur` -0.57; best `salt_pepper` +0.83, `gaussian_noise+salt_pepper` +0.04, `defocus_blur+speckle_noise+salt_pepper` +0.00.

**Verdict: conditioning did not pay off; drop.** Same body as a07 but worse overall (−0.02 dB, DISTS 0.1612 vs 0.1563) and loses a07's unseen advantage (29.08 vs 29.52 dB): the embedding is supervised on the 17 training families, so it does not transfer to new ones. One real win: `salt_pepper` 38.00 dB (best by 0.8 dB). Matches all-in-one literature (AirNet, PromptIR), where conditioning gains are small once the backbone is trained on mixed data.
