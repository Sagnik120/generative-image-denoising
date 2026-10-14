# a20 Conditional GAN (RGB discriminator)

Round 4: strictly generative models. Plan: `../PLAN.md`.

**How it works:** a conditional GAN.
- **Generator** (`src/common/gennet.py::GenNet`): the a05/a15 NAF U-Net (width 24, enc [2,2,2,4], mid 3, dec [1,1,1,2], residual output). After every NAF block a `NoiseInjection` adds one channel of Gaussian noise times a learned per-channel strength (StyleGAN idea). The strength starts at zero, so the network decides how much randomness to use. In eval mode the noise comes from fixed seeded buffers, so the output is reproducible. **4.98 GFLOPs, 5.52 M parameters, one pass.**
- **Discriminator** (`src/common/gan.py::UNetDiscriminatorSN`, training only, zero inference FLOPs): U-Net with spectral normalisation (Real-ESRGAN), 32 base channels, one real/fake score per pixel. Conditional: it sees the corrupted input next to the candidate.
- **Discriminator input:** candidate + corrupted image in RGB (6 channels, 256 px).
**Losses:** composite loss (Charbonnier 1.0 + FFT 0.05 + SSIM 0.2 + DISTS 0.1 from 50%) + hinge adversarial loss, weight 0.005, ramped in over the first 5k steps (on from step 0). Discriminator: hinge loss + R1 gradient penalty every 16 steps.
**Why it was tried:** the cheapest strictly generative model: one pass, discriminator free at inference. a03 (round 1) was the only earlier GAN and was a weak build (L1 only, 3-layer PatchGAN, old data); this is the standard modern recipe on the proven backbone. It is also the baseline that a21's wavelet idea is measured against.
**What to look for:** PSNR/SSIM cost against a15; whether DISTS improves; whether diversity is above zero (if the learned noise strengths stay at zero, the GAN is ignoring its noise); medical and low-light results (adversarial artifacts).
**Data and schedule:** degradation pipeline v2 (22 families), three domains, batch 16, AdamW 4e-4 cosine, EMA 0.999, mixed precision, 60k steps, trained from scratch (no a19 weights).

## Results
Benchmark v1 (396 images, 33 cases), 60k steps, single seed. Diversity = mean per-pixel std over 8 noise seeds, grey levels of 255 (`generative_eval.json`). Reference a15 = same backbone, data and steps, regression.

| | PSNR | SSIM | DISTS | GFLOPs | Diversity | unseen_v2 PSNR |
|---|---|---|---|---|---|---|
| a20 | 31.886 | 0.8554 | 0.1534 | 4.98 | **0.066** | 30.48 |
| a15 (ref.) | 32.034 | 0.8594 | 0.1537 | 4.98 | 0 (deterministic) | 30.99 |
| difference | −0.15 dB | −0.004 | −0.0003 | 0 | | −0.51 dB |

| Group | single | extra | pair | triple | unseen | natural | lowlight | medical | plan (v2) |
|---|---|---|---|---|---|---|---|---|---|
| PSNR | 34.61 | 34.71 | 30.37 | 29.21 | 31.53 | 28.46 | 36.75 | 33.89 | 35.98 |
| vs a15 | −0.28 | +0.18 | −0.15 | −0.14 | −0.08 | −0.12 | −0.19 | −0.16 | −0.23 |

Mean learned noise strength after training: **0.0017** (started at 0). Generator adversarial loss ≈ 0.43, discriminator ≈ 1.47 at the end (hinge; 2.0 = discriminator guessing).

## Analysis
- **The GAN ignored its noise.** The learned noise strengths stayed almost at zero, and 8 seeds differ by 0.07 grey levels on average: invisible, and far below the 0.5 eligibility threshold. a20 is a deterministic network in practice, so it is **not eligible** as a strictly generative model.
- **Why:** every term of the composite loss compares one output with one target. For any random output, the expected error = (error of the mean)² + variance, so the variance is pure penalty. Gradient descent therefore pushes the noise strengths to zero. The adversarial term at weight 0.005 is too weak to pay for any variance, and nothing in the loss rewards different outputs for different seeds. This is the known mode-collapse of conditional GANs (Pix2Pix reported the same; Isola et al., 2017).
- **Cost of the adversarial term itself is small:** −0.15 dB, −0.004 SSIM, equal DISTS vs a15. Largest losses on `chromatic_aberration` −0.60 and `jpeg` −0.58; gain on `small_artifacts` +0.80. So the discriminator setup (U-Net, spectral norm, hinge, R1, ramp) trains stably and does not damage medical or low-light images (−0.16, −0.19 dB): the a03 failure is not repeated.
- **Generalisation is weaker:** `unseen_v2` −0.51 dB vs a15.
- **What it teaches:** a GAN on this backbone costs ~0.15 dB at weak adversarial weight. To make it generative the training must **stop penalising variance** (reconstruction loss on the mean of several samples) and **reward differences between samples** (diversity-sensitive term). See round 5 in `../PLAN.md`.

**Verdict: stable and close to regression quality, but not generative (diversity 0.07). Not eligible. Its recipe is the base for a24.**
