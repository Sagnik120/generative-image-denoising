# a29 Decoder-only, per-pixel gated noise (Option L)

Round 7. Plan: `../PLAN.md` §4. Why: `../COMPARISON.md` §11.3.

**How it works:** a27_full with one change in where and how the noise enters the generator.
- **Noise only after the 5 decoder blocks** (32, 64, 128, 256, 256 px). Encoder (10 blocks) and bottleneck (3) are noise-free, so the skip connections carry clean features.
- **Per-pixel gate:** noise added = `s_c · sigmoid(conv1×1(x)) · n`. s_c is the learned per-channel strength (init 0.01); the gate is computed from the block's own features (init ≈ 0.5); n is one channel of N(0, 1) per pixel.
- Everything else as a27_full: wavelet-domain discriminator, losses, data v3, 16 + 4 batch, 200k steps.
**4.98 GFLOPs (4.982; the five 1×1 gates add ~0.005), 5.52 M parameters, one pass.**

**Why it was tried:** forcing the noise to matter costs ~0.7 dB even without an adversarial term (a28 phase 1). Models with noise in the encoder path lose 1–4 dB on structured corruptions (`small_artifacts`, `periodic_noise`, `salt_pepper`, blurs), and the same variation is added to near-clean images (`resize_blur` below the input). StyleGAN (Karras et al., CVPR 2019) puts noise only in the synthesis path to create stochastic detail without changing structure. Uncertainty in vision outputs is input-dependent (heteroscedastic; Kendall & Gal, NeurIPS 2017), so the variation should be placed per pixel, not uniformly.
**Is it generative?** Yes: noise input, measured diversity (constraint held at 1.5 grey levels during training), adversarial training from step 0.
**What to look for:** PSNR / SSIM vs a27_full at the same epoch; recovery on structured cases; `resize_blur` and `jpeg` at or above the input; `train_spread255` ~1.5; sample-difference maps look like texture, not blotches.

## Results
Benchmark v1, v2, v3 check set; 200k steps, single seed. Run time 25.3 h. Best checkpoint ≈ last epoch (stable curve).

| Noise scale | PSNR | SSIM | DISTS | Diversity |
|---|---|---|---|---|
| **×0.5 (best)** | **32.269** | **0.8682** | **0.1360** | **0.86** |
| ×0.6 | 32.185 | 0.8676 | 0.1355 | 1.00 |
| ×0.75 | 32.069 | 0.8668 | 0.1350 | 1.17 |
| ×1 (as trained) | 31.926 | 0.8655 | 0.1346 | 1.40 |
| ×1.5 | 31.847 | 0.8635 | 0.1352 | 1.61 |
| a27_full ×0.6 (best eligible) | 32.184 | 0.8659 | 0.1444 | 0.675 |
| a25_full ×0.5 | 32.024 | 0.8576 | 0.1399 | 1.04 |
| a19 (regression) | 33.112 | 0.8743 | 0.1293 | 0 |

| Group (×1) | single | extra | pair | triple | unseen | natural | lowlight | medical | unseen_v2 | plan | v3 check |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PSNR | 34.63 | 34.40 | 30.52 | 29.38 | 31.40 | 28.70 | 36.49 | 33.82 | 30.91 | 35.45 | 33.45 |
| DISTS (domains) | | | | | | **0.1382** | **0.1603** | **0.1018** | 0.1829 | 0.0794 | **0.1124** |

**Training curve (last epochs):**

| Epoch | 41 | 44 | 49 | 59 | 69 | 75 | 80 |
|---|---|---|---|---|---|---|---|
| val PSNR | 32.13 | 32.16 | 32.00 | 31.85 | 31.91 | 31.92 | 31.93 |
| val DISTS | 0.1543 | 0.1471 | 0.1444 | 0.1380 | 0.1358 | 0.1348 | **0.1346** |
| `train_spread255` | 1.97 | 2.08 | 1.88 | 1.71 | 1.67 | 1.66 | 1.65 |

## Analysis
- **Best generative model and best eligible operating point: ×0.5 = 32.27 / 0.8682 / 0.1360, diversity 0.86.**
  - Beats a25_full (×0.5) on all three metrics: +0.25 dB, +0.011 SSIM, −0.004 DISTS.
  - Beats a27_full at its best eligible scale (×0.6) on all three: +0.09 dB, +0.002 SSIM, −0.008 DISTS.
  - Gap to the regression a19 is down to −0.84 dB / −0.006 SSIM / +0.007 DISTS (a25_full: −1.09 / −0.017 / +0.011).
- **The gate does what it was designed for.** Training stayed stable: spread held at 1.65–2.1 (target 1.5), with no blow-up like a27_full's 4.5. PSNR settled at 31.93 at ×1 while DISTS kept improving to the last epoch, so the last checkpoint is the best; no selection effect.
- **Best DISTS of all generative models in every domain** (natural 0.138, low-light 0.160, medical 0.102) and on v3 (0.112). The decoder noise produces detail that DISTS rewards.
- **Noise scale matters more than for a25_full:** ×1 → ×0.5 gains +0.34 dB, and diversity stays 0.86. Same lesson as round 6: halving the variance is cheap in diversity and worth PSNR.
- **Weaknesses (at ×1):**
  - `unseen_v2` 30.91, 0.5 dB below a27_full and 0.4 below a25_full (`line_dropout` 36.74 vs 37.53 / 38.13; `film_grain` 27.68 vs 28.50).
  - Structured cases still behind a25_full (`small_artifacts` 39.36 vs 40.59, `chromatic_aberration` 37.24 vs 38.79, `salt_pepper` 38.16 vs 39.34), so the decoder-only noise did **not** fix them; it **did** help blur (`motion_blur` 31.69 vs 31.06 and 31.23).
  - `resize_blur` still below its input (33.56 vs 34.24); `jpeg` +0.23 over the input.
  - All at ×1; at ×0.5 everything moves up by ~0.3 dB.
- **FLOPs** 4.982 vs 4.978: the per-axis rank table puts it behind the 4.978 models on FLOPs, but 0.004 GFLOPs is irrelevant against other teams.

**Verdict: final architecture. Base for hyperparameter tuning and the tuned 200k run (`../HYPERPARAMETER_TUNING.md`). Submission setting so far: noise ×0.5.**
