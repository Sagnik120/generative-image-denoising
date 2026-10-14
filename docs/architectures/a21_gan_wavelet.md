# a21 Conditional GAN with wavelet-domain discriminator and loss

Round 4: strictly generative models. Plan: `../PLAN.md`.

**How it works:** a conditional GAN.
- **Generator** (`src/common/gennet.py::GenNet`): the a05/a15 NAF U-Net (width 24, enc [2,2,2,4], mid 3, dec [1,1,1,2], residual output). After every NAF block a `NoiseInjection` adds one channel of Gaussian noise times a learned per-channel strength (StyleGAN idea). The strength starts at zero, so the network decides how much randomness to use. In eval mode the noise comes from fixed seeded buffers, so the output is reproducible. **4.98 GFLOPs, 5.52 M parameters, one pass.**
- **Discriminator** (`src/common/gan.py::UNetDiscriminatorSN`, training only, zero inference FLOPs): U-Net with spectral normalisation (Real-ESRGAN), 32 base channels, one real/fake score per pixel. Conditional: it sees the corrupted input next to the candidate.
- **Discriminator input:** the three high-frequency Haar bands (horizontal, vertical, diagonal detail) of the candidate and of the corrupted image: 18 channels at 128 px. Noise, blur and compression artifacts live in these bands. Colour and brightness errors are invisible to this discriminator and are left to the composite loss.
**Losses:** composite loss (Charbonnier 1.0 + FFT 0.05 + SSIM 0.2 + DISTS 0.1 from 50%) + L1 on Haar sub-bands of output vs target, weight 0.05 + hinge adversarial loss, weight 0.005, ramped in over the first 5k steps (on from step 0). Discriminator: hinge loss + R1 gradient penalty every 16 steps.
**Why it was tried:** Korkmaz, Tekalp & Dogan (CVPR 2024, "Training Generative Image Super-Resolution Models by Wavelet-Domain Losses Enables Better Control of Artifacts") report that a wavelet-domain discriminator and fidelity loss reduce invented texture and give a better PSNR-vs-perceptual balance. Both parts are training-only, so FLOPs equal a20. The only difference from a20 is the discriminator input and the wavelet loss, so a21 minus a20 is the value of the wavelet idea.
**What to look for:** PSNR/SSIM above a20 at equal or better DISTS; colour or brightness drift (the discriminator cannot see it).
**Data and schedule:** degradation pipeline v2 (22 families), three domains, batch 16, AdamW 4e-4 cosine, EMA 0.999, mixed precision, 60k steps, trained from scratch (no a19 weights).

## Results
Benchmark v1, 60k steps, single seed. Diversity over 8 seeds in grey levels of 255.

| | PSNR | SSIM | DISTS | GFLOPs | Diversity | unseen_v2 PSNR |
|---|---|---|---|---|---|---|
| a21 | 31.809 | 0.8551 | 0.1544 | 4.98 | **0.064** | 30.71 |
| a20 (RGB discriminator) | 31.886 | 0.8554 | 0.1534 | 4.98 | 0.066 | 30.48 |
| a15 (regression ref.) | 32.034 | 0.8594 | 0.1537 | 4.98 | 0 | 30.99 |

| Group | single | extra | pair | triple | unseen | natural | lowlight | medical | plan (v2) |
|---|---|---|---|---|---|---|---|---|---|
| PSNR | 34.50 | 34.78 | 30.30 | 29.17 | 31.37 | 28.30 | 36.71 | 33.93 | 35.86 |
| vs a20 | −0.11 | +0.08 | −0.08 | −0.05 | −0.16 | −0.16 | −0.04 | +0.04 | −0.12 |

Mean learned noise strength 0.0015. Generator adversarial loss drifted to ≈ −0.03 and discriminator loss rose to ≈ 1.63 by the end.

## Analysis
- **Same failure as a20: noise ignored** (diversity 0.06). Not eligible.
- **The wavelet idea was worth nothing measurable on v1:** −0.08 dB, −0.0004 SSIM, +0.001 DISTS vs a20, all at or near single-seed noise (~0.05 dB). Per case it helps `resize_blur` (+0.65 vs a20) and `jpeg` (+0.31) and hurts the noise cases (`salt_pepper` −0.31, `speckle_noise` −0.26) and `periodic_noise` (−0.33).
- **One positive sign:** `unseen_v2` +0.24 dB over a20 (30.71 vs 30.48), mostly the families it never saw. A discriminator that looks only at detail bands may push the generator toward cleaner high frequencies in general. Single seed, 96 images: a hint, not a result.
- **Why the test was weak:** at adversarial weight 0.005 with variance already collapsed, the discriminator barely shapes the generator in either variant; the a21 − a20 gap is mostly the extra wavelet L1 term. The discriminator losses agree: in a21 the discriminator wins (generator loss < 0), i.e. the wavelet discriminator is easy to satisfy and gives weak signal. The wavelet discriminator can only be judged properly in a GAN whose adversarial term actually matters.

**Verdict: no gain at this adversarial strength; not eligible. Kept as a frequency variant (a25) only in combination with the diversity fix, where the discriminator has real influence.**
