# Hyperparameter tuning plan (10-18 → 10-19)

Written 2026-10-16, **revised 2026-10-18 after round 7**. Companion to `PLAN.md` §6. **Implemented 2026-10-18: `scripts/tune_optuna.py`, configs `a29_base_30k`, `a29_tuned_30k`, `a29_tuned` (`WALKTHROUGH.md`).**

Model being tuned: **a29_gan_decnoise** (final architecture, `architectures/a29_gan_decnoise.md`): 32.27 / 0.8682 / 0.1360 at noise ×0.5, diversity 0.86.

---

## 1. Is the plan sound?

The plan: tune a29 with Optuna on 10-18, confirm the best setting at 30k steps, train it for 200k steps on 10-19 → 10-20, then choose, export and submit on 10-20 (deadline 23:59).

**Yes, with five conditions:**

1. **Short trials are biased** ("short-horizon bias", Wu et al., ICLR 2018). A setting that wins after 10k steps does not always win after 200k. Mitigation: every trial runs a complete compressed cosine schedule; the winner is confirmed at 30k steps against the unchanged a29 setting before the 200k run.
2. **Expected gain is modest: +0.1–0.3 dB.** a29's main weaknesses (structured corruptions, unseen families) are not obviously hyperparameter-shaped. Our current values were never tuned, so there is something to find.
3. **The tuned 200k run must beat a29**, which already exists, at its best eligible noise scale (`PLAN.md` §5). Otherwise a29 is submitted. Nothing on 10-20 depends on the tuned run finishing.
4. **Every trial must stay generative, stable and within FLOPs:** diversity ≥ 0.6 at noise ×0.5; sample spread at the end ≤ 3× the target (the a27_full failure mode); FLOPs unchanged (no architecture size change is tuned).
5. **Time is now the binding constraint.** Since 10-17 the machine runs ~1220 s per 2.5k steps (round 6: ~940 s), so a 200k run takes ~27 h. The search is cut to fit (§4).

---

## 2. Method

| Choice | Setting | Why |
|---|---|---|
| Library | **Optuna** (Akiba et al., KDD 2019) | standard, small code, parallel workers through one SQLite study file |
| Sampler | **TPE** (Bergstra et al., NeurIPS 2011), 6 random start-up trials, the current a29 setting enqueued as trial 0 | efficient for 30–40 trials; the baseline anchors the comparison |
| Pruner | **Successive halving / ASHA** (Li et al., Hyperband, JMLR 2018; ASHA, MLSys 2020): rungs at 2.5k and 5k steps, keep the top third | stops weak trials after ~20–45 minutes, so ~2–3× more trials fit |
| Trials | **Optuna, cap 40 trials in total (both workers) or 14 h per worker, whichever comes first**; expected ~35–45 started, ~8–12 run to the end | budget-bound, not count-bound |
| Trial length | **10k steps = 4 epochs of 2.5k**, cosine 4e-4-style schedule compressed to 10k, warm-up 500, DISTS loss from 50% (5k) as in full runs | ~1.36 h per complete trial at the current speed |
| Data | v3, as a29; same data seed in every trial | removes one source of noise between trials |
| Validation | 132-image subset of benchmark v1 (4 per case, all 33 cases, all three domains) + diversity on 16 images × 4 seeds, **at noise ×0.5** (the submission setting) | ~5 min per evaluation instead of ~15 |
| Workers | 2 (one per GPU, one trial per GPU; ~14–17 GB each) | two trials do not fit on one 24 GB GPU |

**Objective (maximise):** `J = PSNR + 20 · SSIM − 20 · DISTS` on the validation subset at noise ×0.5.
- The weights put the three quality metrics on a similar scale: from our results, 0.01 SSIM ≈ 0.01 DISTS ≈ 0.2 dB in how they move together (e.g. a25 → a25_full: +1.07 dB, +0.019 SSIM, −0.026 DISTS).
- FLOPs is fixed by the search space, so it is not in J.

**Constraints (trial marked failed, J = −∞):** diversity < 0.6 at ×0.5; final `train_spread255` > 4.5 (3× target); non-finite losses.

---

## 3. What can be tuned in our model, and what is chosen

### 3.1 Tuned (5 dimensions; cut from 8 to fit the time)

| # | Hyperparameter | Current (a29) | Range | Why it is worth tuning |
|---|---|---|---|---|
| 1 | Peak learning rate (generator) | 4e-4 | log-uniform 2e-4 – 8e-4 | usually the most influential setting; inherited from the regression runs |
| 2 | Discriminator / generator learning-rate ratio | 1 | {0.5, 1, 2} | two-time-scale update rule (TTUR, Heusel et al., NeurIPS 2017); a27_full's late instability came from the generator–discriminator balance |
| 3 | Adversarial weight | 0.005 | log-uniform 0.0025 – 0.01 | moved DISTS and PSNR between rounds 4–6 |
| 4 | Diversity target (grey levels) | 1.5 | 1.0 – 1.8 | the main PSNR ↔ diversity trade-off; a29 at ×0.5 has 0.86, so a lower target is possible with margin |
| 5 | SSIM weight | 0.2 | 0.1 – 0.35 | SSIM is a scored axis; a17/a18 showed loss weights move the PSNR/SSIM/DISTS balance |

### 3.2 Fixed (not tuned), with reasons

| Hyperparameter | Value | Why not tuned |
|---|---|---|
| Width (channels) | 24 | sets FLOPs, a scored axis; the size study (a12/a13) already measured the trade-off |
| **Block layout / number of layers** | enc [2,2,2,4], mid 3, dec [1,1,1,2] | **dropped from the earlier plan.** Changing it also changes how many decoder noise points a29 has, needs new FLOP checks and code, and 10k-step trials cannot judge depth reliably. Too risky in the time left |
| Diversity weight | 4.0 | held the target well in a29 (spread 1.65–2.1) |
| FFT and wavelet-L1 weights | 0.05 / 0.05 | frequency losses showed only small effects (a21/a25 ±0.1 dB on v1); dropped to keep 5 dimensions |
| Batch size | 16 images + 4 extra draws | memory-bound (~14–17 GB); changing it changes time and steps |
| Optimiser, weight decay, warm-up, schedule shape, EMA 0.999, grad clip 1.0 | as now | small expected effect; EMA and long-schedule effects cannot be measured in 10k steps |
| DISTS weight and start (0.1 from 50%) | as now | its effect appears only late (a19, a29 curves after epoch 41) |
| R1 γ and interval | 1, every 16 | stabiliser; changing it risks unstable trials |
| Noise placement and gate | a29 | architecture decision (round 7) |
| Data pipeline | v3 | studied in rounds 3 and 7; expensive to evaluate |
| Test-time noise scale | ×0.5 for now | free; re-checked on the final model (grid 0.4–0.75) |

### 3.3 Not taken, with reasons

| Option | Why not |
|---|---|
| **Dropout** | NAFNet and other restoration networks use no dropout. It removes information the model needs for pixel-exact reconstruction and costs PSNR. Our noise injection is already a stochastic regulariser, and the models do not overfit (validation PSNR rises or plateaus to the end). |
| Width / channel search, kernel size, activation, normalisation | changes FLOPs or the proven NAF block; cannot be judged in 10k steps |
| Grid search | 5 dimensions × 3 values = 243 runs; impossible. TPE + pruning is the efficient choice |
| Population-based training | needs many parallel workers; we have 2 GPUs |

---

## 4. Timeline and compute

At the current machine speed (~1220 s per 2.5k steps); times are approximate.

| Time | GPU 0 | GPU 1 |
|---|---|---|
| 10-18 ~09:00–12:00 | code: `scripts/tune_optuna.py`, subset benchmark, 2-trial smoke test (500 steps each) | |
| 10-18 ~12:00 → 10-19 ~02:00 | **Phase A: Optuna**, worker 1 | **Phase A**, worker 2 |
| 10-19 ~02:00 → ~06:30 | **Phase B:** best trial at 30k steps (~4.3 h) | **Phase B:** a29 baseline setting at 30k steps (same compressed schedule) |
| 10-19 ~07:00 → 10-20 ~10:30 | **tuned 200k run, 80 epochs** (~27 h at current speed, ~21 h if the machine is back to normal) | evaluation of a29 at a finer noise grid (0.4, 0.5, 0.6); export rehearsal; CPU test |
| 10-20 | evaluate the tuned run; final choice; export; report; **submit by 23:59** | |

**Phase A budget:** ~14 h × 2 GPUs = 28 GPU-hours. A complete trial costs ~1.45 h; with pruning the average is ~0.6–0.8 h. That gives **~35–45 trials started and ~8–12 complete** at 10k steps: enough for TPE on 5 dimensions, not on 8.

**Decision rules:**
- **After phase B:** the 200k run uses the tuned setting only if it beats the a29 baseline at 30k by ≥ 0.1 in J, with diversity ≥ 0.6 at ×0.5 and a stable spread. Otherwise no tuned run is made: a29 is final, and 10-19/20 go to export, checks and the report.
- **Run length: 80 epochs (200k steps), decided 10-18.** Checked on a29's curve: over the last 12 epochs (68 → 80) PSNR moved +0.02 dB, DISTS −0.001 and the training loss stayed flat at ~0.049 (learning rate ~1e-6); a19 gained +0.006 dB in its last 9. PSNR peaked at epoch 44 and the second half traded a little PSNR for DISTS (the DISTS loss starts at 50%). So 80 epochs is not cutting the model short: the schedule reaches its plateau. A longer schedule (e.g. 100 epochs) would give at most ~+0.05–0.15 dB by the log-linear trend (60k → 200k = +1.07 dB), and at the current speed it would end ~17:30 on 10-20, too close to the deadline.
- **Why phase B also re-runs the a29 setting at 30k:** a29's existing epoch-12 numbers come from a 200k schedule that was still at a high learning rate (~3.8e-4). A 30k run with its own compressed schedule ends fully annealed. Comparing the tuned 30k run with a29's epoch 12 would credit the tuned setting with the schedule's gain. Both run in parallel on the two GPUs, so it costs no wall time.
- **On 10-20:** the tuned model replaces a29 only if it is better at its best eligible noise scale (`PLAN.md` §5).

---

## 5. Sources

- Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework", KDD 2019.
- Bergstra, Bardenet, Bengio & Kégl, "Algorithms for Hyper-Parameter Optimization" (TPE), NeurIPS 2011.
- Li et al., "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization", JMLR 2018; Li et al., "A System for Massively Parallel Hyperparameter Tuning" (ASHA), MLSys 2020.
- Wu, Ren, Liao & Grosse, "Understanding Short-Horizon Bias in Stochastic Meta-Optimization", ICLR 2018.
- Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (TTUR), NeurIPS 2017.
- Chen et al., "Simple Baselines for Image Restoration" (NAFNet), ECCV 2022.

(Cited from memory; check venues before the report.)
