# a27_full a27 at 200k steps on data v3

Round 7. Plan: `../PLAN.md` §3–4.

**How it works:** exactly a27 (`a27_gan_wave_varctl.md`).
- **Generator:** NAF U-Net width 24, enc [2,2,2,4], mid 3, dec [1,1,1,2], residual output; `NoiseInjection` after **every** NAF block (18), per-channel strength init 0.01. **4.98 GFLOPs, 5.52 M parameters, one pass.**
- **Discriminator** (training only): conditional U-Net with spectral norm on Haar detail bands (LH, HL, HH; 18 ch, 128 px).
- **Batch:** 16 images + a second noise draw for 4 of them (20 outputs).

**Losses:** composite (Charbonnier 1.0 + FFT 0.05 + SSIM 0.2 + DISTS 0.1 from 100k) and wavelet L1 0.05 on every output; diversity hinge (1.5/255, weight 4.0) on the 4 pairs; hinge adversarial 0.005 from step 0 (5k ramp); discriminator hinge + R1. **Frequency:** wavelet-domain discriminator + wavelet fidelity + FFT term.
**Changes from a27:** 200k steps (80 × 2.5k); **data v3**: v2's 22 families + `demosaic_artifacts`, `chroma_noise`, `mri_ghosting`, `ct_ring_artifact`; 12% clean inputs (v2: 8%).
**Why it was tried:** a27 was the round-6 winner (+0.47 dB over a25 at 60k, best noise scale); 60k → 200k was worth +1.07 dB for a25. The low-risk final candidate.
**What to look for:** PSNR vs a25_full (32.02 at ×0.5); diversity ≥ 0.6 at the chosen noise scale; `resize_blur` / `jpeg` vs their inputs; v3 check set.

## Results
Benchmark v1 (396 images), v2, v3 check set; 200k steps, single seed. Run time 25.3 h (machine ~22% slower from epoch 19 on, both runs). **Numbers come from the best checkpoint (selected by composite score on benchmark v1, ~epoch 41–48), not the last epoch.**

| Noise scale | PSNR | SSIM | DISTS | Diversity |
|---|---|---|---|---|
| ×0.5 | 32.225 | 0.8660 | 0.1445 | 0.585 (below the 0.6 margin) |
| ×0.6 (best eligible) | 32.184 | 0.8659 | 0.1444 | 0.675 |
| ×0.75 | 32.125 | 0.8657 | 0.1441 | 0.80 |
| ×1 (as trained) | 32.038 | 0.8653 | 0.1439 | 0.98 |
| a25_full ×0.5 (ref.) | 32.024 | 0.8576 | 0.1399 | 1.04 |

| Group (×1) | single | extra | pair | triple | unseen | natural | lowlight | medical | unseen_v2 | plan | v3 check |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PSNR | 34.88 | 34.49 | 30.57 | 29.41 | 31.49 | 28.76 | 36.72 | 33.91 | **31.42** | 35.74 | 33.50 |

**Training curve (last epochs):**

| Epoch | 41 | 44 | 49 | 59 | 69 | 75 | 80 |
|---|---|---|---|---|---|---|---|
| val PSNR | 32.15 | 32.11 | 31.68 | 31.70 | 31.60 | 31.44 | **31.32** |
| val DISTS | 0.1561 | 0.1489 | 0.1433 | 0.1396 | 0.1433 | 0.1464 | 0.1485 |
| `train_spread255` | 2.35 | 2.72 | 1.74 | 1.68 | 2.86 | 4.04 | **4.48** |

## Analysis
- **The training became unstable in the second half.** After the DISTS loss started (epoch 41), the sample spread repeatedly broke away from the 1.5 target and ended at **4.5 grey levels (3× the target)**. Validation PSNR fell **0.83 dB** from its peak and DISTS rose from 0.140 to 0.149. The diversity hinge only sets a minimum; nothing caps the variance. With noise in all 18 blocks, the generator answered the discriminator by adding more randomness everywhere. The epoch 33–35 wobble was the first sign.
- **Its reported numbers depend on checkpoint selection.** The best checkpoint was picked on the same benchmark v1 that is reported; the last epoch is 0.7 dB lower. A retrain (as planned for 10-19) would not reliably reproduce it.
- **At its best checkpoint it is good:** 32.18 dB at ×0.6, +0.16 dB and +0.008 SSIM over a25_full ×0.5, but DISTS +0.005 worse. Best `unseen_v2` of the generative models (31.42; a25_full 31.30), driven by `film_grain` 28.50 (a25_full 27.53).
- **Diversity margin is thin:** 0.585 at ×0.5 (fails the 0.6 margin), 0.675 at ×0.6.
- **v3 check set:** 33.50 dB / 0.9146 / 0.119; the new families are handled (`chroma_noise` 37.45, `demosaic_artifacts` 36.77, `ct_ring_artifact` 35.55; hardest `mri_ghosting` 32.21).
- **Mild corruptions still over-processed:** `resize_blur` 33.77 (input 34.24); `jpeg` 35.62 (+0.48 over the input, better than a25_full's +0.06).

**Verdict: not chosen. Good best checkpoint, but the recipe is unstable over 200k steps, so it is not a safe base for tuning and retraining.**
