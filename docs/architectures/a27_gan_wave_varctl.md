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
Pending. Fill in from `results/a27_gan_wave_varctl/metrics/`. Goes to 200k if PSNR ≥ 31.33 (a25 + 0.4) with DISTS ≤ 0.172.
