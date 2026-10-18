# Plan: round 7, the final long runs

Written 2026-10-16, after round 6 (a25_full, a27, a28). Replaces the round-6 plan. Background: `COMPARISON.md` §11 (round-6 numbers), `PROJECT_REVIEW_AND_OPTIONS.md` (Options L and M are new), `CURRENT_STATUS.md`.

**Status 2026-10-19: Optuna search done (38 trials; `HYPERPARAMETER_TUNING.md` §5). Decision (confirmed 10-19): train **both** trial 32 (`a29_tuned`, GPU 0) and trial 37 (`a29_tuned_b`, GPU 1) for the full 80 epochs in parallel, **no early stopping**; 30k confirmation skipped for time. Final submission = best of a29 / a29_tuned / a29_tuned_b on 10-20; report written after the choice. Code for the trial-specific configs after approval.**

Hardware: two RTX A5000 (24 GB). Timeline (agreed 10-16, shifted 10-17): **10-17/18** two 200k runs (architecture decision 10-18 morning); **10-18/19** hyperparameter tuning with Optuna (`HYPERPARAMETER_TUNING.md`); **10-19/20** final 200k run with the tuned setting; **10-20** evaluation, export, documentation, submission by 23:59.

---

## 1. What round 6 decided

| Finding (round 6) | Consequence for round 7 |
|---|---|
| a25_full (200k): 32.02 / 0.8576 / 0.1399 at noise ×0.5, diversity 1.04; 3rd of 25 overall; = a15 PSNR, better DISTS and unseen | **Fallback submission.** Already exported; nothing more to train for it |
| 60k → 200k: +1.07 dB for the generative model (regression: +1.08) | Final candidates are trained 200k; 60k results scale predictably |
| a27 (full per-sample loss + diversity constraint): +0.47 dB over a25 at 60k (best noise scale) | **a27_full** (200k): expected ~32.3–32.5 dB |
| a28 (adversarial from 50%) ties a27, −0.48 dB low-light, −0.28 `unseen_v2` | Dropped |
| a28 phase 1: forced diversity costs ~0.7 dB with **no** adversarial term; adding the adversarial term costs nothing | The remaining cost is where and how the noise is injected, not the GAN |
| Every model with noise in the encoder path loses 1–4 dB on structured corruptions (`small_artifacts`, `periodic_noise`, `salt_pepper`, blurs); a20 (noise off) does not | Keep the encoder noise-free (a29) |
| Mild corruptions over-processed: `resize_blur` below the input, `jpeg` ≈ input | Let the network choose per pixel how much variation to add (a29 gate) |
| Noise ×0.5 helps a27/a28 by +0.15/+0.30 dB; their variance is placed where it hurts | Same diagnosis; plus a finer noise-scale grid at selection |

Decisions that stay: strictly generative (noise input, diversity ≥ 0.5, adversarial training from step 0); ~5 GFLOPs, one pass; from scratch; data v2; composite loss; wavelet-domain discriminator; fresh noise per call in the submission.

---

## 2. What to do now: full training, improvement, or tuning?

**Full training of two candidates in parallel; no more screening; tuning only at test time.**

- **No 60k screening round:** with 3 days left, a screened model could not also be trained 200k. Both GPUs run 200k-step candidates that finish on 10-18 morning.
- **GPU 0: a27_full**: the round-6 winner, trained long. Low risk: its recipe is measured.
- **GPU 1: a29_full**: a27 with one targeted change (Option L). Higher risk, higher expected gain. Both runs use the same schedule, so they can be compared epoch by epoch while training. If a29 is behind a27 at the same epoch by more than ~0.2 dB after epoch 30, we know early.
- **Not running:** a28 (dropped), diffusion (closed).
- **Hyperparameter tuning (10-18) and the tuned final run (10-19):** `HYPERPARAMETER_TUNING.md`. Test-time noise scale and the clean-input check are done at final evaluation.

---

## 3. Pending improvements checked (dataset and architecture)

Everything still open from earlier rounds, and whether it goes into the round-7 runs:

| Area | Item | Evidence | Decision |
|---|---|---|---|
| Dataset | **Corruption list v3** (planned since round 3, never done) | adding families was worth +3–8 dB on each added family (round 3); the brief says the private set "may contain other types"; the domains of the private set (low-light camera, MRI, CT) have characteristic artifacts we do not train on | **Include** (§4.1). Both 200k runs use it, so a27_full vs a29_full stays like-for-like |
| Dataset | **Clean / near-clean share** | `resize_blur` output below the input and `jpeg` ≈ input in every generative model: the model changes images that need little change | **Include:** probability of 0 corruptions 8% → 12% (taken from the 1-corruption share, 34% → 30%) |
| Dataset | Add `glass_blur`, `film_grain`, `line_dropout` to training | they are our only measure of generalisation (benchmark v2 `unseen_v2`) | **Not included**: keep them unseen |
| Architecture | Noise placement and gate (a29) | forced noise in the encoder costs ~0.7 dB (round 6) | already in round 7 |
| Architecture | Attention bottleneck, FFT branch, wavelet-domain network, degradation labels | rounds 2–3: all within ±0.1 dB of plain NAF, attention −0.82 dB on unseen (a16) | **Not included** |
| Architecture | Width > 24 | breaks the ~5 GFLOPs budget | **Not included** |
| Architecture | Depth distribution at equal FLOPs | never tested | left to the tuning (`HYPERPARAMETER_TUNING.md`, block layout) |
| Training | Longer than 200k | a19 / a25_full curves flat at the end (schedule exhausted) | **Not included**: no time |

### 3.1 Data v3 (both 200k runs)

**Both a27_full and a29_full use v3**, so their comparison stays like-for-like. What to expect from v3:
- On the 4 new families: large gains, as for every family added in round 3 (+3–8 dB on each).
- On mild corruptions: less over-processing (`resize_blur`, `jpeg`) from the higher clean share.
- On benchmarks v1/v2: about neutral (round 3: +0.1 dB `unseen`, −0 to −0.1 on the rest, because each old family is sampled a little less often). So v3 widens what the model can handle rather than raising every number.
- Its comparison with a25_full (v2 data) mixes the data change with the recipe change; a27_full vs a29_full is the clean architecture comparison.

- Clean images: unchanged (ten sources, natural 70% / medical 20% / low-light, 256×256 crops, flips/rotations, 4% grey).
- Corruptions: v2 (22 families) + **4 new families** (26 total), same chain rules (0–4 per image, severity Beta(1.0, 1.8), random order, 65% of picks from the nine listed families):

| New family | What it simulates | Domain | Severity range (s = 0 → 1) |
|---|---|---|---|
| `demosaic_artifacts` | Bayer CFA sampling + bilinear demosaicing: colour fringes and zipper edges | low-light / natural camera | always on; plus mild noise before demosaicing, σ 0 → 0.02 |
| `chroma_noise` | low-frequency colour blotches (noise in the colour channels only, blurred σ 2–4 px) | low-light | amplitude 0.01 → 0.06 |
| `mri_ghosting` | faint shifted replicas along one axis (phase-encode motion ghosts) | MRI | 1–3 ghosts, intensity 0.05 → 0.25, shift 8–40 px |
| `ct_ring_artifact` | concentric rings around a random centre (detector miscalibration) | CT | amplitude 0.01 → 0.06, ring spacing 3–10 px |

- Probability of 0 corruptions: 12% (v2: 8%).
- Testing: benchmark v1 and v2 unchanged (comparable with every earlier model), plus a **v3 check set**: the 4 new families alone and mixed with Gaussian noise, 12 images each (96 images), seeded.
- Batch: 16 images per step + a second noise draw for 4 of them (20 generator outputs), as a27.

---

## 4. Round-7 models

Shared by both: NAF U-Net generator width 24, enc [2,2,2,4] (256/128/64/32 px), middle 3 (16 px), dec [1,1,1,2] (32/64/128/256 px), residual output. Wavelet-domain conditional U-Net discriminator with spectral norm (Haar LH/HL/HH of candidate and corrupted image, 18 ch, 128 px), training only. AdamW 4e-4, cosine → 1e-6 over **200k steps** (80 × 2.5k), 1k warm-up, EMA 0.999, AMP, grad-clip 1.0, from scratch.

Losses (generator), shared:

| Term | Applied to | Weight |
|---|---|---|
| Composite: Charbonnier 1.0 + FFT 0.05 + (1 − SSIM) 0.2 + DISTS 0.1 (from 50% = step 100k) | every output | 1 |
| Wavelet L1 (Haar sub-bands) | every output | 0.05 |
| Diversity hinge `max(0, 1.5/255 − mean|x1 − x2|)` | the 4 pairs | 4.0 |
| Hinge adversarial | every output | 0.005, from step 0, 5k ramp |

Discriminator: hinge + R1 (γ 1, every 16 steps). **Frequency part:** wavelet-domain discriminator + wavelet fidelity loss + FFT magnitude term.

### a27_full: a27 at 200k (main candidate)

| Item | Setting |
|---|---|
| Model | exactly a27 (`architectures/a27_gan_wave_varctl.md`): `NoiseInjection` after **every** NAF block (18 blocks), per-channel strength, init 0.01 |
| Change | 200k steps instead of 60k; data v3 (§3.1) |
| FLOPs | 4.98 GFLOPs, 5.52 M parameters |
| Expected | ~32.3 dB at ×1, ~32.4–32.5 at the best scale; DISTS ~0.135; diversity ~0.9–1.0 at ×1 |

### a29_gan_decnoise: decoder-only, per-pixel gated noise (Option L)

| Item | Setting |
|---|---|
| Generator | a27's U-Net, but **noise injection only after the 5 decoder blocks** (32, 64, 128, 256, 256 px). Encoder (10 blocks) and bottleneck (3) are noise-free, so the skip connections carry clean features |
| Noise | `x + s_c · g(x) · n`. n: one channel of N(0, 1) per pixel. s_c: learned per-channel strength (init 0.01, as before). **g(x) = sigmoid(1×1 conv(x))**: a learned per-pixel gate from the block's own features (bias init 0, so g = 0.5 at start) |
| Why | structured corruptions are detected in the encoder; noise there blurs what is detected (StyleGAN puts noise only in the synthesis path, Karras et al. 2019). The gate lets the model vary where the true image is uncertain and stay exact where it is not, e.g. near-clean inputs and flat areas (heteroscedastic uncertainty, Kendall & Gal 2017) |
| Losses, discriminator, data (v3), schedule | as a27_full |
| Frequency | as a27_full |
| FLOPs | ~4.99 GFLOPs (five 1×1 convolutions C → 1: ~0.01 GFLOPs), 5.52 M parameters |
| Expected | recover part of the ~0.7 dB forced-diversity cost: ~32.5–32.8 dB; fewer losses on `small_artifacts`, `periodic_noise`, `salt_pepper`; `resize_blur` and `jpeg` at or above the input |
| What to check | `train_spread255` at ~1.5 (the constraint holds); gate maps not collapsed to a few pixels; sample-difference images look like texture, not blotches |
| Is it generative? | yes: noise input, measured diversity, adversarial training from step 0. Only the place where the noise enters changes |

### Fallback

a25_full (done). If both new runs fail, it is the submission at noise ×0.5.

---

## 5. How the model is chosen (architecture on 10-18; final on 10-20)

1. Evaluate a25_full, a27_full, a29_full at noise ×0.5, ×0.6, ×0.75, ×1 (v1, v2, diversity over 8 seeds).
2. **Eligible:** diversity ≥ 0.6 at that scale (margin over 0.5), FLOPs ≤ 5.1.
3. Rank PSNR, SSIM, DISTS separately among the eligible (model, scale) pairs; best average rank wins; tie → better `unseen_v2`.
4. Reject a candidate whose medical or low-light loss vs a19 is > 0.5 dB beyond its overall loss.
5. Clean-input check: on uncorrupted benchmark images the output must not be worse than the input by more than ~0.5 dB.

---

## 6. Round 8: tuned final run and submission (revised 2026-10-19)

### 6.1 What is trained

| Run | GPU | Setting (from Optuna) | Steps | Purpose |
|---|---|---|---|---|
| **a29_tuned** | 0 | **trial 32**: lr 7.55e-4, D/G lr ratio 0.5, adversarial 0.00275, diversity target 1.60, SSIM weight 0.334 | 200k (80 epochs) | main tuned candidate: safest of the top trials (discriminator lr ≈ a29's, best diversity margin, best SSIM) |
| **a29_tuned_b** | 1 | **trial 37**: lr 7.9e-4, D/G ratio 2, adversarial 0.0033, diversity target 1.37, SSIM weight 0.19 | 200k (80 epochs) | best J of the search; riskier (discriminator lr ~1.6e-3, diversity 0.625 at 10k) |

Everything else as a29: network (decoder-only gated noise, 4.98 GFLOPs), wavelet discriminator, losses, data v3, batch 16 + 4 draws, EMA 0.999, cosine schedule with 1k warm-up.

### 6.2 Why straight to 80 epochs (no 30k confirmation)

- **Time:** a 200k run takes ~25 h at the current machine speed. It must start on 10-19 to finish on 10-20 morning, leaving the day for export and the report. A 30k confirmation first (~4.5 h) would push the end to ~10-20 15:00.
- **Risk accepted:** the 10k advantage (+1.0 J, mostly from a higher learning rate) may shrink at 200k (short-horizon bias). The guard is that a29 is already trained: the tuned model is used only if it is better.
- **80 epochs is enough:** a29's curve was flat for its last 12 epochs (`HYPERPARAMETER_TUNING.md` §4).

### 6.3 Monitoring during the run (no early stopping)

Both runs train the full 80 epochs, whatever the curves show; the comparison is made only at the end. The curves are still checked against a29 at the same epoch, to explain the result:

| Check | a29 reference |
|---|---|
| `train_spread255` (targets: 1.60 for trial 32, 1.37 for trial 37) | 1.65–2.1 (target 1.5) |
| val PSNR at epochs 12 / 24 / 40 / 80 (noise ×1) | 30.97 / 31.66 / 32.08 / 31.93 |
| val DISTS at epochs 44 / 80 | 0.147 / 0.135 |

Since the best checkpoint is chosen by composite score on benchmark v1, a run whose curve collapses late (as a27_full) still reports its best epoch. That number is flagged as "best checkpoint, not last" in the comparison.

### 6.4 Final choice (10-20)

Noise-scale evaluation needs almost no extra time on 10-20: `run_all.py` already runs `eval_generative.py` at the end of each training, at noise ×0.5 / 0.6 / 0.75 / 1 / 1.5 (benchmark v1 + diversity over 8 seeds; ~10–12 min per scale, ~50–60 min per model, included in the ~25 h). So when the runs finish, the full noise-scale table for both tuned models is ready, and a29's already exists. Optional extras on 10-20:
- noise ×0.4, only if a model still has diversity ≥ 0.75 at ×0.5 (~12 min per model; both GPUs in parallel);
- benchmarks v2/v3 and the group/domain breakdown at the chosen scale for the report (~20 min for the final model; needs a `--noise_scale` option in `evaluate.py`, a small code step).

1. **Eligible:** diversity ≥ 0.6 at that scale; FLOPs ≤ 5.1 GFLOPs.
2. **Rank** PSNR, SSIM and DISTS separately among the eligible (model, scale) pairs; best average rank wins. Tie → better `unseen_v2`, then better low-light and medical.
3. **Reference:** a29 at ×0.5 = 32.27 / 0.8682 / 0.1360, diversity 0.86.
4. Fallbacks: a29, then a25_full.

### 6.5 Schedule

| When | GPU 0 | GPU 1 | Other |
|---|---|---|---|
| 10-19 morning | code: `--write_configs --trial N` (~30 min, after approval), dry run | – | – |
| 10-19 ~11:00 → 10-20 ~12:00 | **a29_tuned (trial 32), 80 epochs** | **a29_tuned_b (trial 37), 80 epochs** | export rehearsal of a29 on CPU (fresh noise per call, uint8, fvcore + ptflops) |
| 10-20 ~12:00 → 13:00 | final choice from the automatic noise-scale tables (§6.4); optional ×0.4 and breakdown at the chosen scale (~30 min) | (parallel) | |
| 10-20 13:00 → 23:59 | export of the final model (~15 min), CPU test from a clean checkout (~15 min), diversity sample images | | **report (1–2 pages), written after the choice; submission by 23:59** |

Memory: ~14–17 GB per run (as a29). Disk: ~0.3 GB per run on `/DATA1/shrusti/genai_project/results`.

---

## 7. Code (round 7 implemented 2026-10-17; round 8 tuning code 2026-10-18; small step pending)

- `gennet.py`: `NoiseInjection` option `gate: true` (1×1 conv C → 1 + sigmoid, bias 0); `GenNet` option `noise_where: all | decoder` (inject only in decoder blocks).
- Configs: `a27_full` (a27 config, 80 epochs), `a29_gan_decnoise` (a27_full + `noise_where: decoder`, `noise_gate: true`); registry; architecture docs.
- `eval_generative.py`: default noise scales 0.5 0.6 0.75 1.0 1.5 (argument already exists).
- `degradations.py`: 4 new families and the `v3` family set; 0-corruption probability in the configs (12%); v3 check set in the benchmark builder.
- Round 8 (done): `scripts/tune_optuna.py` (Optuna TPE + successive halving, 2 workers, SQLite study, 132-image subset at noise ×0.5); `gan.disc_lr_ratio` in `gan.py`; configs `a29_base_30k` (phase-B baseline), `a29_tuned_30k`, `a29_tuned` (written from the best trial by `--write_configs`); `optuna` in `requirements.txt`.
- Pending (10-19, after approval): `scripts/tune_optuna.py --write_configs --trial N --arch <name>` to write the config of a chosen trial (not only the best), and a registry entry `a29_tuned_b` for trial 37.
