# a29_tuned_b: a29 with Optuna trial 37 hyperparameters

Round 8. Same network, losses and data as a29_tuned (`a29_tuned.md`); hyperparameters of Optuna trial 37 (best J in the search): lr **7.91e-4**, discriminator lr = **2** × generator lr, adversarial **0.00334**, diversity target **1.37**, SSIM weight **0.189**. 200k steps, 4.98 GFLOPs.

## Results
Run time 25.25 h. Stable curve (spread 1.61–1.79); best checkpoint = last epoch.

| Noise scale | PSNR | SSIM | DISTS | Diversity |
|---|---|---|---|---|
| ×0.5 | 32.713 | 0.8709 | 0.1353 | 0.379 |
| ×0.6 | 32.692 | 0.8707 | 0.1350 | 0.438 |
| ×0.75 | 32.658 | 0.8703 | 0.1346 | 0.522 |
| **×1 (only eligible setting ≤ ×1)** | **32.601** | **0.8697** | **0.1340** | **0.649** |
| ×1.5 | 32.482 | 0.8683 | 0.1341 | 0.876 |

| Group (×1) | single | extra | pair | triple | natural | lowlight | medical | unseen_v2 | plan | v3 check |
|---|---|---|---|---|---|---|---|---|---|---|
| PSNR | 35.72 | 35.35 | 30.98 | 29.71 | 29.07 | 37.43 | **34.83** | 31.93 | 36.53 | 34.02 |
| DISTS (domains) | | | | | 0.138 | 0.158 | 0.101 | 0.178 | 0.074 | 0.112 |

## Analysis
- **Almost the same quality as a29_tuned at ×1** (−0.01 dB, −0.004 SSIM, **−0.002 DISTS**; best DISTS of all generative models, 0.1340). Better on medical (+0.34 dB) and `chromatic_aberration` (39.80).
- **But its generative margin is thin:** the lower diversity target (1.37) and the stronger discriminator (lr ratio 2) gave diversity 0.649 at ×1 and below 0.6 at every lower scale. It can only be submitted at ×1, where it is 0.14 dB / 0.004 SSIM behind a29_tuned at ×0.75, with diversity just above the bar.
- Confirms the search: the diversity target ≳ 1.4 is needed for a safe margin after 200k steps.

**Verdict: not chosen. Same quality class as a29_tuned, but its diversity margin (0.65 at ×1) is too thin for a strict "generative" claim, and it is behind on PSNR and SSIM at a29_tuned's chosen setting.**
