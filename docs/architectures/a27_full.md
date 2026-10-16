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
Pending. Fill in from `results/a27_full/metrics/` (`final_evaluation_summary.json` incl. `benchmark_v3`, `generative_eval.json`).
