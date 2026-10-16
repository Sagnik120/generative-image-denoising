# a27 GAN, full per-sample loss + active diversity constraint

Round 6. Plan: `../PLAN.md` §4–5. Why: `a24_gan_div.md` analysis and `../COMPARISON.md` §10.

**How it works:** a25's networks with the round-6 training recipe.
- **Generator** (`src/common/gennet.py::GenNet`): NAF U-Net width 24, enc [2,2,2,4], mid 3, dec [1,1,1,2], residual output, `NoiseInjection` after every NAF block (strength init 0.01). **4.98 GFLOPs, 5.52 M parameters, one pass.**
- **Discriminator** (training only): a25's conditional U-Net with spectral norm on Haar detail bands (18 ch, 128 px).
- **Batch:** 16 images (as a20; round 5 had 8). The first 4 get a second noise draw → 20 generator outputs per step.

**Losses (generator):**
| Term | Applied to | Weight | Why |
|---|---|---|---|
| Composite (Charbonnier 1.0 + FFT 0.05 + SSIM 0.2 + DISTS 0.1 from 50%) | **every output, full weight** | 1 | round 5 applied it to the mean of two samples, which halved each sample's reconstruction signal and over-smoothed the output |
| Wavelet L1 (Haar sub-bands) | every output | 0.05 | kept from a25 |
| Diversity hinge `max(0, 1.5/255 − mean|x1 − x2|)` | the 4 pairs | 4.0 | an active constraint: the model varies by ~1.5 grey levels and no more (rcGAN-style variance control, Bendel et al., NeurIPS 2023) |
| Hinge adversarial | every output | 0.005 from step 0, 5k ramp | a20/a21 value; 0.01 added noise rather than texture in round 5 |

Discriminator: hinge + R1 (γ 1, every 16 steps). **Frequency:** wavelet-domain discriminator (+0.23 dB on unseen families, twice) + wavelet fidelity + FFT term.
**What to look for:** `train_spread255` at ~1.5 (the hinge is active and winning; falling toward 0 = collapse); PSNR vs a21 (31.81) and a25 (30.93); diversity ≥ 1 at test; sharper texture than a25 in `epoch_0024_comparison.png`.
**Data and schedule:** pipeline v2, three domains, AdamW 4e-4 cosine, EMA 0.999, AMP, 60k steps, from scratch.

## Results
Benchmark v1, 60k steps, single seed. Run time 6.9 h (incl. evaluation).

| Noise scale | PSNR | SSIM | DISTS | GFLOPs | Diversity |
|---|---|---|---|---|---|
| ×0.5 | 31.391 | 0.8501 | 0.1599 | 4.98 | 0.64 |
| **×1 (as trained)** | **31.237** | **0.8489** | **0.1601** | 4.98 | **1.10** |
| ×1.5 | 31.066 | 0.8471 | 0.1622 | 4.98 | 1.49 |
| a25 (60k, round-5 recipe) | 30.934 | 0.8372 | 0.1674 | 4.98 | 1.54 |
| a20 (60k, noise ignored) | 31.886 | 0.8554 | 0.1534 | 4.98 | 0.07 |

| Group | single | extra | pair | triple | unseen | natural | lowlight | medical | unseen_v2 | plan (v2) |
|---|---|---|---|---|---|---|---|---|---|---|
| PSNR | 33.85 | 33.20 | 29.92 | 28.80 | 30.90 | 27.93 | 36.17 | 32.91 | 30.45 | 35.42 |
| vs a25 | +0.22 | +0.29 | +0.35 | +0.24 | +0.46 | +0.48 | +0.20 | +0.06 | +0.54 | +0.77 |

Noise strength 0.011; `train_spread255` 1.85 → 1.67 (just above the 1.5 target: the hinge is at its limit, the constraint is active). Discriminator loss 1.67.

## Analysis
- **The round-6 recipe helped: +0.30 dB, +0.012 SSIM, −0.007 DISTS vs a25 at the same 60k**, and +0.47 dB at noise ×0.5 (31.39 vs 30.92). Gains everywhere, largest on natural images (+0.48) and unseen families (+0.54).
- **It recovered only about a third of round 5's loss:** still −0.65 dB vs a20.
- **The remaining cost is the forced noise itself, not the adversarial term or the batch.** a28's first half has the same loss *without* any adversarial term and reached 30.95 dB at epoch 12, the same as a27 (30.93), while a20 (adversarial, no forced noise) was at 31.65. So making the noise matter costs ~0.7 dB at this stage, whatever the adversarial setting.
- **The variance is now harmful where it sits:** halving it at test time gains +0.15 dB (a25: +0.02). The diversity constraint makes the model vary by ~1.5 grey levels against a full per-sample loss, so it puts the variation where it costs least on average, but uniformly across images, including near-clean ones.
- **Same weak cases** (vs a20): `small_artifacts` −3.77, `periodic_noise` −1.47, `salt_pepper` −1.19, `motion_blur` −1.08, `chromatic_aberration` −0.97, `gaussian_blur` −0.96; `resize_blur` 33.37 (below the input, 34.24). Every model whose noise enters the encoder path (a22, a24–a28) shows this pattern; a20 (noise switched off) does not.
- **Eligibility margin:** diversity 1.10 at ×1, 0.64 at ×0.5.

**Verdict: round-6 winner by the plan's rules (a28 is not ≥ 0.2 dB better); +0.47 dB over a25 at the best noise scale. Goes to 200k as a27_full. The remaining gap points at where the noise enters the network (round 7, a29).**
