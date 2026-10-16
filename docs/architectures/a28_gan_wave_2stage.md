# a28 a27 with the adversarial term delayed (two-phase training)

Round 6. Plan: `../PLAN.md` §5.

**How it works:** identical networks, batch and losses to a27 (`a27_gan_wave_varctl.md`). Only the schedule of the adversarial term differs (`gan.adv_start_frac: 0.5`):
- **Phase 1 (steps 0–30k):** composite + wavelet L1 on every output + diversity constraint (1.5/255, weight 4.0). The noise is in use and enforced from step 0. No adversarial term and no discriminator training (also faster).
- **Phase 2 (30k–60k):** a27's full objective: hinge adversarial 0.005 ramped over 5k steps, discriminator hinge + R1. DISTS also starts at 30k (50%, as in every model).
**4.98 GFLOPs, 5.52 M parameters, one pass. Frequency:** wavelet-domain discriminator (phase 2) + wavelet fidelity + FFT term.
**Why it was tried:** SRGAN (Ledig et al., CVPR 2017), ESRGAN (Wang et al., ECCVW 2018) and Real-ESRGAN (Wang et al., ICCVW 2021) all train for reconstruction first and then adversarially; that is how GAN restorers keep high PSNR. a28 − a27 measures whether this beats adversarial training from step 0 on our data.
**Is it generative?** Yes: it samples throughout training (noise input, diversity enforced from step 0) and the final half is adversarial training, which is the distribution-matching objective. The models above are called GANs with no noise input at all.
**What to look for:** PSNR at epoch 12 (end of phase 1, should be close to a20's 31.65 at that point) and the drop or gain after the switch; `train_spread255` at ~1.5 throughout; final PSNR vs a27.
**Data and schedule:** as a27.

## Results
Benchmark v1, 60k steps, single seed. Run time 6.4 h (incl. evaluation; phase 1 has no discriminator).

| Noise scale | PSNR | SSIM | DISTS | GFLOPs | Diversity |
|---|---|---|---|---|---|
| ×0.5 | 31.451 | 0.8524 | 0.1580 | 4.98 | 0.74 |
| **×1 (as trained)** | **31.154** | **0.8511** | **0.1582** | 4.98 | **1.28** |
| ×1.5 | 30.892 | 0.8493 | 0.1598 | 4.98 | 1.71 |
| a27 (adversarial from step 0) | 31.237 | 0.8489 | 0.1601 | 4.98 | 1.10 |

| Group | single | extra | pair | triple | unseen | natural | lowlight | medical | unseen_v2 | plan (v2) |
|---|---|---|---|---|---|---|---|---|---|---|
| PSNR | 33.65 | 33.16 | 29.86 | 28.84 | 30.82 | 28.01 | 35.70 | 32.90 | 30.17 | 34.87 |
| vs a27 | −0.20 | −0.04 | −0.06 | +0.05 | −0.07 | +0.08 | **−0.48** | −0.01 | −0.28 | −0.56 |

Phase 1 (epochs 1–12, no adversarial term): 30.95 dB at epoch 12 (a27: 30.93; a20: 31.65). Phase 2: +0.20 dB over the last 12 epochs (a27: +0.31). Noise strength 0.006; `train_spread255` 1.86 → 1.64.

## Analysis
- **The two-phase schedule did not help:** −0.08 dB vs a27 at ×1, +0.06 at ×0.5; +0.002 SSIM, −0.002 DISTS. A tie on v1, below the ≥ 0.2 dB rule.
- **It is worse where it matters for the private set:** low-light −0.48 dB, `unseen_v2` −0.28. Starting the adversarial term late gave it less time to shape the low-light and unseen behaviour.
- **Most useful as an experiment:** its phase 1 isolates the cost of forced diversity without any adversarial term (−0.7 dB vs a20 at epoch 12). The adversarial term is not what costs PSNR; it even helps DISTS.
- Still clearly generative (diversity 1.28; 0.74 at ×0.5) despite the delayed adversarial term.

**Verdict: not taken forward. a27's simpler schedule is kept (adversarial from step 0, better low-light and unseen results).**
