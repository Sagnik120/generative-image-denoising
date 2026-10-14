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
Pending. Fill in from `results/a25_gan_div_wave/metrics/`. Reference: a21 31.809 / 0.8551 / 0.1544, `unseen_v2` 30.71.
