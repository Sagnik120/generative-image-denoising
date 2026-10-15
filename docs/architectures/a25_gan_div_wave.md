# a25 a24 with wavelet-domain discriminator and wavelet loss

Round 5. Plan: `../PLAN.md` §5.

**How it works:** identical to a24 (generator, two-sample batch, stochastic recipe, adversarial 0.01) except for two training-only changes, the same as a21 over a20:
- **Discriminator input:** the three high-frequency Haar bands (LH, HL, HH) of the candidate and of the corrupted image, 18 channels at 128 px. Noise, blur and compression artifacts live there; colour and brightness are left to the composite loss.
- **Wavelet fidelity loss:** L1 between Haar sub-bands of the mean sample and the target, weight 0.05.
**4.98 GFLOPs, 5.52 M parameters, one pass** (both changes cost nothing at inference).

**Losses:** a24's (composite on the mean of 2 samples + 0.2 Charbonnier per sample + diversity hinge, target 2/255 + hinge adversarial 0.01) + 0.05 wavelet L1. **Frequency:** wavelet-domain adversarial + wavelet fidelity (Korkmaz, Tekalp & Dogan, CVPR 2024) + FFT term in the composite loss.
**Why it was tried:** a21 showed no gain on v1 (−0.08 dB) but +0.24 dB on unseen families, at an adversarial weight that barely mattered. With the adversarial term now doing real work, a25 − a24 is a fair measure of the wavelet discriminator.
**What to look for:** PSNR / DISTS vs a24; `unseen_v2`; colour or brightness drift (the discriminator cannot see the low band); diversity.
**Data and schedule:** as a24.

## Results
Benchmark v1, 60k steps, single seed. Run time 5.5 h (incl. evaluation).

| Noise scale | PSNR | SSIM | DISTS | GFLOPs | Diversity |
|---|---|---|---|---|---|
| ×0.5 | 30.921 | 0.8392 | 0.1643 | 4.98 | 1.13 |
| **×1 (as trained)** | **30.934** | **0.8372** | **0.1674** | 4.98 | **1.54** |
| ×1.5 | 30.893 | 0.8344 | 0.1762 | 4.98 | 1.70 |
| a24 (RGB discriminator) | 30.907 | 0.8349 | 0.1666 | 4.98 | 1.69 |

| Group | single | extra | pair | triple | unseen | natural | lowlight | medical | unseen_v2 | plan (v2) |
|---|---|---|---|---|---|---|---|---|---|---|
| PSNR | 33.63 | 32.91 | 29.57 | 28.56 | 30.44 | 27.45 | 35.97 | 32.86 | 29.91 | 34.66 |
| vs a24 | +0.15 | −0.14 | +0.04 | −0.05 | −0.08 | −0.01 | +0.18 | −0.05 | **+0.23** | +0.08 |

Noise strength 0.0198; `train_spread255` 2.3 → 3.1 (epoch 8) → 2.8; discriminator loss ~1.86 (closer to chance than a24's 1.73).

## Analysis
- **Eligible (1.54) and marginally the best round-5 model:** +0.03 dB, +0.0023 SSIM, +0.0008 DISTS vs a24, i.e. a tie on v1. Best avg rank of round 5 (15.5 vs 16.0).
- **The wavelet discriminator helps unseen corruptions, consistently:** `unseen_v2` +0.23 dB over a24, the same as a21 over a20 (+0.24). Two independent pairs, same sign and size, so this is the most reliable small effect found in rounds 4–5.
- **Per case** (vs a24) it helps the noise cases (`salt_pepper` +0.79, `correlated_noise` +0.36, `poisson_noise` +0.34, `chromatic_aberration` +0.33, `gaussian_noise` +0.27) and hurts deblurring and periodic patterns (`periodic_noise` −0.71, `small_artifacts` −0.51, `motion_blur` −0.39). Logical: its discriminator sees only detail bands, where residual noise is obvious, while blur and periodic patterns also involve the low band and the phase structure it cannot see.
- **Same over-smoothing as a24** (−0.88 dB vs a21); the wavelet change does not touch the cause.
- 20% faster training than a24 (the discriminator works at 128 px).

**Verdict: best eligible model so far (30.93 / 0.8372 / 0.1674, diversity 1.54). Its wavelet discriminator and wavelet loss are carried into round 6.**
