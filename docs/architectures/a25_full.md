# a25_full a25 trained 200k steps (compliant fallback)

Round 6. Plan: `../PLAN.md` §2, §5.

**How it works:** exactly a25 (`a25_gan_div_wave.md`): NAF U-Net generator (width 24, enc [2,2,2,4], mid 3, dec [1,1,1,2]) with noise injection after every block (init 0.01); conditional U-Net discriminator with spectral norm on the Haar detail bands (LH, HL, HH) of candidate and corrupted image. **4.98 GFLOPs, 5.52 M parameters, one pass.**
**Losses:** composite (Charbonnier 1.0 + FFT 0.05 + SSIM 0.2 + DISTS 0.1 from 50%) on the mean of 2 samples + 0.2 Charbonnier per sample + diversity hinge (2/255, weight 1.0) + wavelet L1 0.05 + hinge adversarial 0.01 (5k ramp); discriminator hinge + R1. **Frequency:** wavelet-domain discriminator + wavelet fidelity + FFT term.
**Only change:** 200k steps (80 epochs × 2.5k) instead of 60k; cosine schedule and DISTS start (100k) scale with it. 8 images × 2 draws per step.
**Why it was tried:** a25 is the best eligible model (30.93 / 0.8372 / 0.1674, diversity 1.54). For regression, 60k → 200k gave +1.08 dB (a15 → a19). This run gives a submittable model whatever round 6 shows.
**What to look for:** PSNR gain over a25 (expected ~+1 dB, to ~31.9); diversity still ≥ 0.5 after long training; `unseen_v2`.

## Results
Pending. Fill in from `results/a25_full/metrics/`. Reference: a25 (60k) 30.934 / 0.8372 / 0.1674, diversity 1.54, `unseen_v2` 29.91; a19 (regression, 200k) 33.112 / 0.8743 / 0.1293.
