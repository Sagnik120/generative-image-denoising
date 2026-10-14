# a26 One-step residual diffusion + GAN

Round 5. Plan: `../PLAN.md` §5. Code: `src/common/resdiff.py::ResDiffGANBundle` (training) on top of `src/common/gan.py::GANBundle`.

**How it works:** a22's residual diffusion model with an adversarial loss on its one sampling step (Denoising Diffusion GANs, Xiao, Kreis & Vahdat, ICLR 2022).
- **Forward process** (ResShift, Yue et al., NeurIPS 2023), T = 4: `x_t = x0 + η_t (y − x0) + κ √η_t ε`, η = (t/4)², κ = 0.1. At the last level `x_T = y + 0.1 ε`.
- **Network** (`ResDiffNet`): NAF U-Net width 24, 6-channel input (`y`, `x_t`), time embedding by FiLM, output = clean-image estimate.
- **Training levels:** 75% at the last level (the only level the 1-step sampler uses; a22: 50%), 25% at random levels. Both noise draws of one image share the level.
- **Sampler:** start at `y + κ ε` (ε drawn inside the call, seeded in eval), one pass. **5.02 GFLOPs, 5.68 M parameters.**
- **Discriminator:** a24's conditional RGB U-Net, judging the clean-image estimate next to the corrupted image (training only).

**Losses (generator):** composite on the mean of 2 estimates (two ε draws) + 0.2 Charbonnier per estimate + diversity hinge (target 2/255, weight 1.0) + hinge adversarial 0.01 (5k ramp). Discriminator: hinge + R1. **Frequency:** FFT term in the composite loss.
**Why it was tried:** a22 has diversity by construction but paid −0.52 dB for noise it then erased, because nothing rewarded using it. The adversarial and diversity terms make the noise useful; the level weighting gives the 1-step path more training. It is the most clearly "generative" model of the round: a diffusion forward process, a diffusion training objective and a GAN-modelled reverse step.
**What to look for:** diversity ≥ 1; PSNR between a22 (31.51) and a24; whether the structured cases (`small_artifacts`, `salt_pepper`, `periodic_noise`) recover from a22's −1 to −2 dB; noise-scale sweep.
**Data and schedule:** pipeline v2, 8 images × 2 draws per step, 60k steps, from scratch.

## Results
Pending. Fill in from `results/a26_resdiff_gan/metrics/`. Reference: a22 31.511 / 0.8499 / 0.1654, diversity 0.50.
