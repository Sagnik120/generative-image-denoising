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
Pending. After the run, fill in from `results/a20_gan/metrics/final_evaluation_summary.json` (PSNR / SSIM / DISTS / GFLOPs, groups, domains, benchmark v2) and `generative_eval.json` (diversity and mean noise strength). Reference: a15 (regression, same data and steps) 32.03 dB / 0.8594 / 0.1537; `unseen_v2` 30.99 dB.
