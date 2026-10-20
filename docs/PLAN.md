# Plan: final model and submission (2026-10-20)

Written 2026-10-20, after round 8. Replaces all earlier plans; their reasoning is kept in `COMPARISON.md` §9–13, `HYPERPARAMETER_TUNING.md`, `PROJECT_REVIEW_AND_OPTIONS.md` and `architectures/`. **Deadline: today, 23:59.**

**Status: final model chosen. Remaining: small export/check code step (after approval), evaluation at the chosen noise scale, export, checks, report, submission.**

---

## 1. Final model

**a29_tuned** (`architectures/a29_tuned.md`), submitted at **test-time noise scale ×0.75**.

| | Value |
|---|---|
| PSNR / SSIM / DISTS (benchmark v1, 396 images) | **32.741 / 0.8740 / 0.1371** |
| FLOPs (fvcore, 1×3×256×256) | **4.98 GFLOPs**, one forward pass, 5.52 M parameters |
| Diversity (std over 8 seeds, grey levels) | **0.77** |
| Unseen families (`unseen_v2`, ×1) | 32.15 dB (best of all models, incl. the regression a19's 31.88) |
| v3 check set (×1) | 33.97 dB / 0.9198 / 0.112 |
| Gap to the regression a19 (not allowed) | −0.37 dB, −0.0003 SSIM, +0.008 DISTS |

**Model.** A conditional GAN.
- **Generator:** a NAF U-Net (width 24; encoder [2,2,2,4], bottleneck 3, decoder [1,1,1,2]; residual output). Gaussian noise is injected after each of the 5 decoder blocks, scaled by a learned per-channel strength and a learned per-pixel gate. At inference the noise is drawn inside the model call and multiplied by 0.75.
- **Discriminator** (training only, no inference FLOPs): conditional U-Net with spectral norm on Haar wavelet detail bands.

**Training objective.**
- Composite reconstruction loss on every sample: Charbonnier 1.0 + FFT 0.05 + (1 − SSIM) 0.334 + DISTS 0.1 from mid-training, plus wavelet L1 0.05.
- Hinge adversarial loss 0.00275 with R1 on the discriminator.
- Diversity constraint: two samples of the same input must differ by ≥ 1.6 grey levels.
- AdamW, lr 7.55e-4 (discriminator 0.5×), cosine schedule, EMA 0.999, 200k steps, batch 16 (+4 extra noise draws), from scratch.
- Hyperparameters from a 38-trial Optuna search.

**Data.**
- ~80k clean images from 10 public research datasets, in three domains: natural (DIV2K, Flickr2K, BSDS, COCO), low-light (LOL plus synthetic darkening), and medical (chest X-ray, IXI MRI, CT, ultrasound).
- 26 corruption families, chained 0–4 per image at random severities and order, with 12% clean inputs:
  - the brief's nine;
  - 13 extra: correlated, Rician, stripe, sensor and uniform noise, random impulses, resize blur, pixelate, WebP, quantisation, ringing, double JPEG, small artifacts;
  - 4 domain-specific: demosaicing, chroma noise, MRI ghosting, CT rings.

**Why this one** (`COMPARISON.md` §13):
- **Best of the eligible candidates on PSNR and SSIM.** It ties a29_tuned_b on average rank and wins the tie-break on unseen families.
- **Larger generative margin.** a29_tuned_b's diversity is 0.65 at ×1 and below 0.6 at every lower scale.
- **Why ×0.75:** it keeps diversity at 0.77, a 29% margin over 0.6. ×0.6 would add only +0.08 dB, with diversity 0.64.

**Fallbacks:** a29_tuned_b at ×1, then a29 at ×0.5.

---

## 2. What the brief requires (`Mini Competition.pdf`) and how we meet it

| § | Requirement | Our submission | Check |
|---|---|---|---|
| Title, 1 | A **small generative** model that denoises and removes small artifacts; general-purpose | Conditional GAN with a noise input (different noise → different restoration; diversity 0.77), adversarial objective, 4.98 GFLOPs; 26 corruption families, 3 domains | diversity measured; samples shown in the report |
| 2 | Input one corrupted RGB 256×256×3; output 256×256×3, **uint8 [0, 255]**; code converts if needed | `denoise_image(uint8 array) → uint8 array`; clamp, ×255, round, uint8 | export verification + `check_submission.py` |
| 3.1 | Training data we have the right to use | Public research datasets (DIV2K, Flickr2K, BSDS500, COCO, LOL, NIH ChestX-ray, IXI, CT and ultrasound sets) | listed with sources in the report |
| 3.2 | Broad, varied corruptions and severities | 26 families, random chains and severities, mixed orders | listed in the report |
| 4 | ~100 private images: natural, low-light, medical; undisclosed degradations | All three domains in training and in our benchmarks; unseen-family test (`unseen_v2`) | reported |
| 5 | FLOPs by fvcore / ptflops on (1, 3, 256, 256) | One pass; the noise is drawn inside `forward`; FLOPs of the module returned by `load_model()` | **fvcore and ptflops**, both reported |
| 5 | PSNR (skimage), SSIM (skimage), DISTS (official DISTS-pytorch) on uint8 | Our evaluation uses the same implementations on uint8 images | same code path as all our numbers |
| 6 | Four axes ranked independently | Balanced design: high PSNR/SSIM, competitive DISTS, small FLOPs | – |
| 7.1 | Weights in a standard format + inference code, 256×256×3 in → 256×256×3 out | `model_weights.pt` (PyTorch `state_dict`), `inference.py` (`load_model()`, `denoise_image()`, CLI), the model code (`src/`), `model_config.json`, `README.md` | clean-checkout CPU test |
| 7.2 | Report, 1–2 pages: architecture, training objective, degradations | `docs/REPORT.md` → PDF (outline in §5) | page count |

---

## 3. Code step needed (small; after approval)

1. **`scripts/export_inference.py --noise_scale 0.75`**: the exported `load_model()` applies the noise scale, so both `denoise_image()` and the module measured for FLOPs run exactly the submitted setting. Noise stays fresh per call (`seed=None`); `denoise_image(img, seed=k)` stays reproducible. The scale is written into `model_config.json` and `README.md`.
2. **`scripts/check_submission.py`** (new), run on the exported folder only (as a grader would):
   - load through `inference.py`;
   - FLOPs with **fvcore and ptflops** on (1, 3, 256, 256);
   - output shape and dtype;
   - same seed → identical output; fresh noise → different output (diversity);
   - PSNR / SSIM / DISTS on 33 benchmark images (one per case) as a sanity check;
   - 3 sample images × 4 seeds and their per-pixel std map, saved for the report;
   - CPU time per image.
3. `requirements.txt`: `ptflops`.

---

## 4. Steps today (10-20)

| # | Step | Where | Time |
|---|---|---|---|
| 1 | Approve this plan → code step §3 + CPU test | here | ~45 min |
| 2 | Commit, push, pull on the GPU machine | – | ~5 min |
| 3 | Breakdown at the chosen scale: `evaluate.py --arch a29_tuned --noise_scale 0.75` (v1 groups/domains, v2, v3) | GPU | ~20 min |
| 4 | Export: `export_inference.py --arch a29_tuned --noise_scale 0.75` → `$R/a29_tuned/submission/` | GPU / CPU | ~5 min |
| 5 | Checks: `check_submission.py` on the exported folder, on **CPU** (fresh shell, only the package) | CPU | ~15 min |
| 6 | Report: `docs/REPORT.md` filled with the numbers from steps 3–5 and the sample images → PDF (≤ 2 pages) | here | ~2–3 h |
| 7 | Package: `submission/` (weights, `inference.py`, `src/`, config, README, requirements) + report PDF → zip; final re-check from the zip on CPU | – | ~30 min |
| 8 | Submit | – | target **by ~20:00**, 4 h buffer before 23:59 |

Not done today: no more training, no more noise-scale changes after step 3 (×0.4 is skipped: a29_tuned's diversity at ×0.5 is 0.55, below the 0.75 condition).

---

## 5. Report outline (1–2 pages)

1. **Task and approach** (3–4 lines): a small conditional GAN; one pass; 4.98 GFLOPs; samples via injected noise.
2. **Architecture**, with a figure:
   - NAF U-Net generator; decoder-only gated noise injection;
   - wavelet-domain U-Net discriminator (training only);
   - parameters and FLOPs (fvcore and ptflops).
3. **Training objective:**
   - composite reconstruction loss on every sample;
   - hinge adversarial loss with R1;
   - diversity constraint;
   - EMA, schedule, 200k steps;
   - Optuna tuning (38 trials, 5 hyperparameters).
4. **Degradations and data:** 26 families (the brief's 9, 13 extra, 4 domain-specific) as a compact table; chain and severity sampling; 12% clean; datasets and domains.
5. **Results:**
   - benchmark v1 / v2 / v3 table at ×0.75;
   - per-domain row;
   - generative evidence: diversity and a sample/std figure;
   - ablation path (a25 → a25_full → a29 → a29_tuned);
   - comparison with our regression reference.
6. **Limitations** (2 lines): structured artifacts and `resize_blur` over-processing; in-distribution benchmark.

---

## 6. What to send back after steps 3–5

- `$R/a29_tuned/metrics/final_evaluation_summary_noise0.75.json` and the `_noise0.75` breakdown CSVs.
- The `check_submission.py` output (FLOPs from both counters, timing, metrics) and the saved sample images.
