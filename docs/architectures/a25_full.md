# a25_full a25 trained 200k steps (compliant fallback)

Round 6. Plan: `../PLAN.md` §2, §5.

**How it works:** exactly a25 (`a25_gan_div_wave.md`): NAF U-Net generator (width 24, enc [2,2,2,4], mid 3, dec [1,1,1,2]) with noise injection after every block (init 0.01); conditional U-Net discriminator with spectral norm on the Haar detail bands (LH, HL, HH) of candidate and corrupted image. **4.98 GFLOPs, 5.52 M parameters, one pass.**
**Losses:** composite (Charbonnier 1.0 + FFT 0.05 + SSIM 0.2 + DISTS 0.1 from 50%) on the mean of 2 samples + 0.2 Charbonnier per sample + diversity hinge (2/255, weight 1.0) + wavelet L1 0.05 + hinge adversarial 0.01 (5k ramp); discriminator hinge + R1. **Frequency:** wavelet-domain discriminator + wavelet fidelity + FFT term.
**Only change:** 200k steps (80 epochs × 2.5k) instead of 60k; cosine schedule and DISTS start (100k) scale with it. 8 images × 2 draws per step.
**Why it was tried:** a25 is the best eligible model (30.93 / 0.8372 / 0.1674, diversity 1.54). For regression, 60k → 200k gave +1.08 dB (a15 → a19). This run gives a submittable model whatever round 6 shows.
**What to look for:** PSNR gain over a25 (expected ~+1 dB, to ~31.9); diversity still ≥ 0.5 after long training; `unseen_v2`.

## Results
Benchmark v1 (396 images, 33 cases), 200k steps, single seed. Run time 20.4 h on one RTX A5000 (incl. evaluation).

| Noise scale | PSNR | SSIM | DISTS | GFLOPs | Diversity |
|---|---|---|---|---|---|
| **×0.5 (best)** | **32.024** | **0.8576** | **0.1399** | 4.98 | 1.04 |
| ×1 (as trained) | 32.005 | 0.8566 | 0.1410 | 4.98 | 1.34 |
| ×1.5 | 31.969 | 0.8554 | 0.1455 | 4.98 | 1.46 |
| a25 (60k) | 30.934 | 0.8372 | 0.1674 | 4.98 | 1.54 |
| a15 (regression, 60k) | 32.034 | 0.8594 | 0.1537 | 4.98 | 0 |
| a19 (regression, 200k) | 33.112 | 0.8743 | 0.1293 | 4.98 | 0 |

| Group | single | extra | pair | triple | unseen | natural | lowlight | medical | unseen_v2 | plan (v2) |
|---|---|---|---|---|---|---|---|---|---|---|
| PSNR | 34.94 | 34.81 | 30.40 | 29.20 | 31.61 | 28.45 | 36.76 | 34.36 | 31.30 | 35.71 |
| vs a25 (60k) | +1.31 | +1.90 | +0.83 | +0.64 | +1.17 | +1.00 | +0.79 | +1.50 | +1.39 | +1.05 |
| vs a19 | −1.33 | −2.27 | −0.78 | −0.76 | −1.15 | −1.06 | −1.23 | −1.09 | −0.58 | −1.67 |

Learned noise strength 0.052 (a25: 0.020). `train_spread255` 2.7–2.8 throughout. Validation PSNR +0.05 dB over the last 16 epochs (the cosine schedule has run out, as for a19).

## Analysis
- **Long training paid exactly as for regression:** +1.07 dB, +0.019 SSIM, −0.026 DISTS over a25 (a15 → a19: +1.08 dB). Diversity 1.34 (×1) / 1.04 (×0.5): still clearly generative after 200k steps.
- **Best generative model and 3rd of all 25 models overall** (average rank 5.6, behind a19 and a16). It now **equals the 60k regression model a15 on PSNR (−0.03 dB)**, is **better on DISTS (0.141 vs 0.154, 2nd of all models)** and on unseen families (`unseen_v2` 31.30 vs 30.99).
- **Gap to a19 (regression, same steps): −1.11 dB, −0.018 SSIM, +0.012 DISTS.** This is the price of being generative with this recipe at 200k: about the same 1 dB as at 60k (a25 − a20 + a20 − a15 ≈ −1.1), so long training did not close it.
- **Where the gap to a19 is** (per case): `small_artifacts` −4.87, `periodic_noise` −2.43, `salt_pepper` −2.06, `chromatic_aberration` −1.76, `motion_blur` −1.52, `gaussian_blur` −1.38, `jpeg` −1.23, `resize_blur` −1.24; only −0.4 to −0.8 on the random-noise mixes. The same structured / exact-inverse cases as in round 5.
- **Mild corruptions are over-processed:** `resize_blur` 33.78 dB is **below the input (34.24)** and `jpeg` 35.20 is only +0.06 above it (a19: +1.29). On images that are already near-clean, the model's own sample variance (~1.3 grey levels everywhere) and smoothing cost more than there is to fix.
- **Visual check** (`epoch_0080_comparison.png`): much sharper than a24/a25 at 60k (coral, grass, wave spray kept). Remaining: the finest detail under heavy noise (small spots in the water, the "South Tyne" lettering) is smoothed away. No grain or GAN artifacts.
- **Noise ×0.5 is the best setting** (+0.02 dB, −0.0011 DISTS, diversity 1.04).

**Verdict: current best compliant model and the fallback submission (×0.5: 32.02 / 0.8576 / 0.1399, 4.98 GFLOPs, diversity 1.04).**
