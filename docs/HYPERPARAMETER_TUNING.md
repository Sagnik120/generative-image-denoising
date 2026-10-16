# Hyperparameter tuning plan (10-18)

Written 2026-10-16. Companion to `PLAN.md` (§6 schedule). **Planning only: no code yet.** The tuning code is written on 10-17 while the 200k runs train. Dates updated 10-17: runs start 10-17 midday, so Optuna runs 10-18 ~11:30 → 10-19 ~05:00, phase B 10-19 ~05:00–12:00, tuned 200k run 10-19 ~12:30 → 10-20 ~10:30 (`PLAN.md` §6).

---

## 1. Is the plan sound?

The plan: decide and train the architecture at 200k by 10-17; tune hyperparameters with Optuna on 10-18 using short trials; train the final model with the tuned values on 10-19; documentation and submission on 10-20 (deadline 23:59).

**Yes, with four conditions:**

1. **Short trials are biased.** A setting that wins after 10k steps does not always win after 200k ("short-horizon bias", Wu et al., ICLR 2018). For example, a high learning rate converges fast early and plateaus lower. Mitigation: each trial uses a complete (compressed) cosine schedule, so it converges within its budget; the best 2 settings are then confirmed at 30k against the baseline before the 200k run.
2. **The gain is modest.** Expect +0.1–0.3 dB. Our current values were never tuned (they come from the regression runs), so there is something to find, but nothing like the +1 dB of long training.
3. **The 10-19 run is the last 200k run possible.** It must start by 10-19 ~01:00 to finish (~22 h) before 10-20 morning. If it is late or worse than the 10-17 model at the same step, the 10-17 model is submitted. Nothing on 10-20 depends on it.
4. **Every trial must stay generative and within FLOPs:** a trial whose diversity is below 0.6, or whose architecture exceeds 5.1 GFLOPs, counts as failed.

---

## 2. Method

| Choice | Setting | Why |
|---|---|---|
| Library | **Optuna** (Akiba et al., KDD 2019) | standard, small code, parallel workers through one SQLite study file |
| Sampler | **TPE** (Bergstra et al., NeurIPS 2011), 8 random start-up trials, the current configuration enqueued as trial 0 | efficient for ~30–60 trials and mixed continuous / categorical spaces; the baseline anchors the comparison |
| Pruner | **Successive halving / ASHA** (Li et al., Hyperband, JMLR 2018; ASHA, MLSys 2020): rungs at 2.5k, 5k, 10k steps, keep the top third | stops weak trials early, so ~2–3× more trials fit |
| Trial length | **10k steps**, cosine 4e-4-style schedule compressed to 10k, warm-up 500, DISTS from 50% (5k) as in full runs | ~1.1 h per complete trial at a27's speed |
| Data | pipeline v3 (as the 10-17 runs), same seed per trial for the data stream | removes one source of noise between trials |
| Validation | 132-image subset of benchmark v1 (4 per case, all 33 cases, all three domains) + diversity on 16 images × 4 seeds | ~4 min per evaluation instead of ~15 |
| Workers | 2 (one per GPU, one trial per GPU; ~14–17 GB each) | two trials do not fit on one 24 GB GPU |

**Objective (maximise):** `J = PSNR + 20 · SSIM − 20 · DISTS` on the validation subset.
- Weights put the three quality metrics on a similar scale: from our results 0.01 SSIM ≈ 0.01 DISTS ≈ 0.2 dB in how they move together (e.g. a25 → a25_full: +1.07 dB, +0.019 SSIM, −0.026 DISTS).
- FLOPs is not in J: it is fixed (~5 GFLOPs) by the constraint below.

**Constraints (trial marked failed, J = −∞):** diversity < 0.6 grey levels at noise ×1; FLOPs > 5.1 GFLOPs (checked before training); non-finite losses.

---

## 3. What can be tuned in our model, and what is chosen

### 3.1 Tuned (8 dimensions)

| # | Hyperparameter | Current | Range | Why it is worth tuning |
|---|---|---|---|---|
| 1 | Peak learning rate (generator) | 4e-4 | log-uniform 2e-4 – 1e-3 | usually the most influential setting; inherited from regression, never tuned for the GAN |
| 2 | Discriminator / generator learning-rate ratio | 1 | {0.5, 1, 2} | two-time-scale update rule (TTUR, Heusel et al., NeurIPS 2017); our discriminator sits near chance (loss 1.67–1.87 of 2.0) |
| 3 | Adversarial weight | 0.005 | log-uniform 0.002 – 0.01 | round 4 vs 5: the weight changed DISTS and PSNR |
| 4 | Diversity target (grey levels) | 1.5 | 1.0 – 2.0 | the main PSNR ↔ diversity trade-off; lower costs less PSNR but reduces the eligibility margin |
| 5 | Diversity weight | 4.0 | {2, 4, 8} | must be strong enough to hold the target against the per-sample loss |
| 6 | SSIM weight | 0.2 | 0.1 – 0.4 | SSIM is a scored axis; a17/a18 showed loss weights move the PSNR/SSIM/DISTS balance |
| 7 | FFT and wavelet-L1 weights (frequency losses) | 0.05 / 0.05 | 0.02 – 0.1 / 0 – 0.1 | the frequency part of the objective; affects blur vs sharpness |
| 8 | **Block layout** (the "number of layers") | enc [2,2,2,4], mid 3, dec [1,1,1,2] | 3 predefined layouts, all 4.9–5.1 GFLOPs (exact layouts fixed on 10-17 with fvcore) | how depth is distributed between resolutions, at equal FLOPs |

Layout candidates (to be FLOP-checked on 10-17; only those within 4.9–5.1 GFLOPs are kept):
- L0 current: enc [2,2,2,4], mid 3, dec [1,1,1,2].
- L1 deeper at low resolution: enc [1,2,2,6], mid 4, dec [1,1,1,1].
- L2 heavier decoder: enc [2,2,2,3], mid 2, dec [1,1,2,2] (more capacity where the decoder noise acts, if a29 wins).

### 3.2 Fixed (not tuned), with reasons

| Hyperparameter | Value | Why not tuned |
|---|---|---|
| Width (channels) | 24 | sets FLOPs, which is a scored axis; the size study (a12/a13) already measured the trade-off |
| Batch size | 16 images + 4 extra draws | memory-bound (~14–17 GB of 24); changing it changes steps per epoch and time |
| Optimiser | AdamW, β (0.9, 0.99), weight decay 1e-4 | standard for NAFNet-style restoration; small expected effect |
| Schedule shape, warm-up, EMA 0.999, grad clip 1.0 | as now | small effect, and short trials cannot measure EMA or long-schedule effects reliably |
| DISTS weight and start (0.1 from 50%) | as now | its effect appears only late in training (a19 curve, epochs 41+); studied in a17/a18 |
| R1 γ and interval | 1, every 16 | stabiliser; changing it risks unstable trials for little expected gain |
| Noise placement / gate | from the 10-17 architecture decision | architecture choice, already decided by the full runs |
| Data pipeline (families, probabilities, severity) | v3 | expensive to evaluate; studied in round 3 |
| Test-time noise scale | chosen at evaluation | free, no training needed (grid 0.5–1.0) |

### 3.3 Not taken, with reasons

| Option | Why not |
|---|---|
| **Dropout** | NAFNet and other restoration networks use no dropout. It removes information the model needs to reconstruct pixels exactly and costs PSNR. Our noise injection is already a stochastic regulariser, and with ~50k training images and 200k steps the models do not overfit (validation PSNR rises to the end). |
| Number of channels per layer / width search | changes FLOPs; covered by the size study |
| Kernel sizes, activation, normalisation | NAF block design is fixed and proven in rounds 2–3; changes would need longer than 10k steps to judge |
| Grid search | 8 dimensions × 3 values = 6,561 runs; impossible. TPE + pruning is the efficient choice |
| Population-based training | needs many parallel workers; we have 2 GPUs |

---

## 4. Timeline and compute

| Time | GPU 0 | GPU 1 |
|---|---|---|
| 10-17 (day) | 200k run (architecture) | 200k run (architecture) |
| 10-17 (day, CPU) | write `scripts/tune_optuna.py`, the subset benchmark, layout FLOP checks; 2-trial smoke test | |
| 10-17 ~21:00 → 10-18 ~16:00 | **Optuna phase A**: ~19 h × 2 GPUs. With pruning, ~40–50 trials started, ~12–15 complete | same study, second worker |
| 10-18 16:00 → ~23:30 | **Phase B**: top-1 trial at 30k steps (~3.3 h), then baseline at 30k | top-2 trial at 30k steps (~3.3 h) |
| 10-19 ~00:30 | **final 200k run with the winning setting** (~22 h → 10-19 ~23:00) | evaluation of the 10-17 model at noise scales; export rehearsal |
| 10-20 | evaluate the 10-19 run; choose final (it must beat the 10-17 model on the §5 rules of `PLAN.md`); export; documentation; submit by 23:59 | |

**Decision rule after phase B:** the 200k run uses the tuned setting only if it beats the baseline at 30k by ≥ 0.1 in J and keeps diversity ≥ 0.6. Otherwise the 10-19 GPU time goes to a second seed of the 10-17 configuration or to nothing, and the 10-17 model is final.

---

## 5. Sources

- Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework", KDD 2019.
- Bergstra, Bardenet, Bengio & Kégl, "Algorithms for Hyper-Parameter Optimization" (TPE), NeurIPS 2011.
- Li et al., "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization", JMLR 2018; Li et al., "A System for Massively Parallel Hyperparameter Tuning" (ASHA), MLSys 2020.
- Wu, Ren, Liao & Grosse, "Understanding Short-Horizon Bias in Stochastic Meta-Optimization", ICLR 2018.
- Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (TTUR), NeurIPS 2017.
- Chen et al., "Simple Baselines for Image Restoration" (NAFNet), ECCV 2022.

(Cited from memory; check venues before the report.)
