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
Pending. After the run, fill in from `results/a21_gan_wavelet/metrics/final_evaluation_summary.json` (PSNR / SSIM / DISTS / GFLOPs, groups, domains, benchmark v2) and `generative_eval.json` (diversity and mean noise strength). Reference: a15 (regression, same data and steps) 32.03 dB / 0.8594 / 0.1537; `unseen_v2` 30.99 dB.
