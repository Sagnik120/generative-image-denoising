# Report material: where everything is

For the teammate writing the 1–2 page technical report (brief §7.2: architecture, training objective, degradations used in training). The submitted model is called **"generative denoiser"** in the report and the package. Internally it is `a29_tuned`; do not use internal names (a29, rounds, trials) in the report.

All paths are relative to the repo root.

## 1. Final numbers (submitted setting: test-time noise scale 0.75)

Source: `results/a29_tuned/metrics/final_evaluation_summary_noise0.75.json`.

| Split | PSNR (dB) | SSIM | DISTS |
|---|---|---|---|
| Benchmark v1, all 396 images (input 24.49 dB) | **32.74** | **0.874** | **0.137** |
| Natural (198) | 29.25 | 0.838 | 0.141 |
| Low-light (99) | 37.73 | 0.901 | 0.161 |
| Medical (99) | 34.74 | 0.919 | 0.105 |
| Single corruptions / extra / pairs / triples | 35.82 / 35.84 / 31.08 / 29.81 | | |
| Corruption types never seen in training (v2 `unseen_v2`, film grain, glass blur, line dropout + mixes) | 32.21 | 0.830 | 0.178 |
| Domain-specific check set (v3: demosaicing, chroma noise, MRI ghosting, CT rings) | 34.07 | 0.920 | 0.113 |

| Efficiency | Value | Source |
|---|---|---|
| FLOPs, fvcore, (1, 3, 256, 256) | **4.98 GFLOPs** | package check (`docs/TASK.md` §2); `final_evaluation_summary_noise0.75.json` → `gflops` |
| FLOPs, ptflops | 5.09 GMACs | package check |
| Parameters | 5.52 M | same |
| Forward passes | 1 | – |

**Generative evidence:** mean per-pixel standard deviation over 8 noise seeds = **0.77 grey levels** at ×0.75. The table over noise scales (×0.5 → ×1.5) is in `results/a29_tuned/metrics/generative_eval.json`.

**Per-case tables** (33 corruption cases, input vs output PSNR/SSIM/DISTS):
- `results/a29_tuned/metrics/benchmark_breakdown_noise0.75.csv` (v1);
- `benchmark_v2_breakdown_noise0.75.csv` (v2) and `benchmark_v3_breakdown_noise0.75.csv` (v3), in the same folder.

## 2. Architecture (report section "Architecture")

Text source: `final_submission/generative_denoiser/README.md` ("Model"), `docs/architectures/a29_tuned.md`, code `final_submission/generative_denoiser/model.py`.

- **Conditional GAN.**
- **Generator:** U-Net of NAFNet blocks (Chen et al., ECCV 2022).
  - Width 24.
  - Encoder 2/2/2/4 blocks at 256/128/64/32 px; bottleneck 3 blocks at 16 px; decoder 1/1/1/2 blocks.
  - Residual output: the network predicts a correction to the input.
- **Stochasticity:** Gaussian noise injected after each of the 5 decoder blocks, scaled by a learned per-channel strength (StyleGAN-style) and a learned per-pixel gate (sigmoid of a 1×1 conv). The encoder and bottleneck are noise-free. At test time the noise is multiplied by 0.75.
- **Discriminator** (training only, 1.10 M parameters): conditional U-Net with spectral normalisation. It sees the Haar wavelet detail bands (LH, HL, HH) of the candidate and of the corrupted image, as in Korkmaz et al., CVPR 2024. Its output is a per-pixel real/fake score.
- **Figure:** draw the U-Net with noise+gate symbols on the decoder blocks; sample outputs as below (§5).

## 3. Training objective (report section "Training objective")

Text source: `docs/architectures/a29_tuned.md`, config `final_submission/generative_denoiser/training/src/architectures/generative_denoiser/config.yaml`, code `training/src/common/gan.py`.

- **Reconstruction on every generated sample:** Charbonnier 1.0 + FFT-magnitude 0.05 + (1 − SSIM) 0.334 + DISTS 0.1 (from 50% of training) + Haar-wavelet L1 0.05.
- **Adversarial:** hinge loss, weight 0.00275 (ramped in over 5k steps); discriminator hinge loss + R1 gradient penalty (γ = 1, every 16 steps).
- **Diversity constraint:** a second noise draw for 4 of the 16 images per batch; hinge penalty `4 · max(0, 1.6/255 − mean|x₁ − x₂|)`, so samples differ by about 1.6 grey levels.
- **Optimisation:**
  - AdamW, lr 7.55e-4 (discriminator 0.5×), cosine to 1e-6, 1k warm-up;
  - EMA 0.999 of the generator, mixed precision;
  - batch 16, 200k steps (~25 h on one RTX A5000), trained from scratch.
- **Hyperparameters** chosen with an Optuna search: TPE + successive halving, 38 trials of 10k steps, 5 hyperparameters. Source: `docs/HYPERPARAMETER_TUNING.md` §5, `results/optuna/trials_a29_tune.csv`. One line in the report is enough.

## 4. Data and degradations (report section "Degradations")

Text source: `docs/PROJECT_REVIEW_AND_OPTIONS.md` §1, `docs/PLAN.md` §3 (round-7 plan, data v3), code `training/src/common/degradations.py`, `dataset.py`.

- **Clean images** (~80k, public research datasets):
  - natural: DIV2K, Flickr2K, BSDS500, COCO;
  - low-light: LOL + synthetic darkening;
  - medical: chest X-ray, IXI MRI, CT, ultrasound.

  Sampling ≈ natural 70%, medical 20%, low-light 10%; 256×256 random crops, flips and rotations.
- **26 corruption families:**
  - the brief's nine: Gaussian, Poisson, salt-and-pepper and speckle noise; Gaussian, motion and defocus blur; JPEG; chromatic aberration;
  - 13 extra: correlated, Rician, stripe, sensor and uniform noise; random impulses; resize blur; pixelate; WebP; quantisation; ringing; double JPEG; small artifacts (spots/scratches);
  - 4 domain-specific: demosaicing artifacts, chroma noise, MRI ghosting, CT ring artifacts.
- **Sampling:** 0–4 corruptions per image (12% / 30% / 36% / 17% / 5%); severities from Beta(1.0, 1.8), milder for longer chains; 50% random order; 65% of picks from the nine listed families; 12% clean inputs (so the model learns to leave clean images alone).
- **Benchmarks** (fixed, held-out images):
  - v1: 396 images, 33 cases, 3 domains;
  - v2: 96 images of families never trained on;
  - v3: 96 images of the 4 domain-specific families.

## 5. Figures available

| Figure | Path |
|---|---|
| Input / output / ground truth for 6 natural images (epoch 80) | `results/a29_tuned/epoch_0080_comparison.png` |
| More per-epoch comparisons | `results/a29_tuned/visualizations/` |
| Validation PSNR / SSIM / DISTS curves | `results/a29_tuned/metrics/val_psnr_curve.png`, `val_ssim_curve.png`, `val_dists_curve.png`, `metrics_dashboard.png` |
| Training loss curves | `results/a29_tuned/loss_curves/` |
| Diversity samples (4 seeds + std map; natural, low-light, medical) | not yet made: optional, `docs/TASK.md` §4.2 on the GPU machine |

## 6. Optional context (one short paragraph or a small table)

How the final design was reached. Source: `docs/COMPARISON.md` §13.4. The report can say "ablations", without internal names.

| Step | PSNR | SSIM | DISTS |
|---|---|---|---|
| GAN with a forced-diversity loss, 60k steps | 30.93 | 0.837 | 0.167 |
| + long training (200k) | 32.02 | 0.858 | 0.140 |
| + per-sample loss, more data, decoder-only gated noise | 32.27 | 0.868 | 0.136 |
| + tuned hyperparameters (submitted) | 32.74 | 0.874 | 0.137 |

Limitations (from `docs/architectures/a29_tuned.md`): structured artifacts (small artifacts, salt-and-pepper, periodic noise) are 2–3 dB behind a non-generative regression network of the same size; mild resize blur is slightly over-processed.
