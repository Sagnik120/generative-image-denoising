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
Benchmark v1, 60k steps, single seed. Run time 6.7 h (incl. evaluation).

| Steps / noise scale | PSNR | SSIM | DISTS | GFLOPs | Diversity |
|---|---|---|---|---|---|
| 1 step, ×0.5 | 30.880 | 0.8363 | 0.1764 | 5.02 | 0.97 |
| **1 step, ×1 (as trained)** | **30.637** | **0.8299** | **0.1775** | 5.02 | **1.79** |
| 1 step, ×1.5 | 30.192 | 0.8180 | 0.1862 | 5.02 | 2.60 |
| 2 steps | 29.230 | 0.7916 | 0.2024 | 10.04 | 3.61 |
| 4 steps | 27.714 | 0.7291 | 0.2285 | 20.08 | 5.71 |
| a22 (ref., same diffusion, no GAN) | 31.511 | 0.8499 | 0.1654 | 5.02 | 0.50 |

| Group | single | extra | pair | triple | unseen | natural | lowlight | medical | unseen_v2 | plan (v2) |
|---|---|---|---|---|---|---|---|---|---|---|
| PSNR | 33.14 | 32.42 | 29.36 | 28.57 | 30.07 | 27.27 | 35.52 | 32.48 | 28.46 | 34.46 |
| vs a22 | −1.07 | −1.18 | −0.74 | −0.53 | −1.04 | −0.81 | −1.00 | −0.87 | −1.27 | −1.94 |

`train_spread255` 2.5–2.7 throughout; validation PSNR flat from epoch 20 (30.64 → 30.62).

## Analysis
- **Eligible (1.79) but the weakest round-5 model:** −0.87 dB vs a22, −0.27 vs a24, worst DISTS of the round (0.1775). Last in the overall ranking of all 22 models apart from a10.
- **Same cause as a24** (halved per-sample reconstruction gradient, 8 images per step, stronger adversarial weight on a weak discriminator), on top of a22's own cost of carrying input noise. Its per-case losses repeat a22's pattern, deeper: `small_artifacts` −2.26, `periodic_noise` −1.70, `chromatic_aberration` −1.34, `salt_pepper` −1.11 vs a22.
- **The noise is not used well:** halving it at test time *gains* +0.24 dB with DISTS almost unchanged; ×1.5 loses 0.45 dB. Unlike the GANs, a26's variance is mostly harm.
- **More steps are still much worse** (−1.4 dB at 2 steps, −2.9 at 4), with diversity rising to 5.7: each extra pass adds noise the model cannot convert into detail.
- **Generalisation is poor:** `unseen_v2` 28.46, −1.23 below a24.

**Verdict: rejected. The diffusion line (a22, a23, a26) is closed: at ≤ 5 GFLOPs it is behind the GAN on every metric in every round.**
