# a04 Tiny conditional DDPM (SR3-style)

**How it works:** diffusion from pure noise conditioned on the corrupted image, T=8 steps, noise-prediction MSE. FLOPs = per-step × 8.
**Observation:** collapsed (never converged); started from noise instead of the corrupted image, motivating a10.

## Results
Round 1 (old pipeline, not comparable with round 2): PSNR 10.01, SSIM 0.0861, DISTS 0.5223, 37.33 GFLOPs. Discard.
