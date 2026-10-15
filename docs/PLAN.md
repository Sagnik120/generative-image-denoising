# Plan: round 6, recover the lost dB, and the final long run

Written 2026-10-15 (Day 2 of 6), after round 5 (a24–a26). Replaces the round-5 plan. Background: `COMPARISON.md` §9–10 (round 4–5 numbers), `PROJECT_REVIEW_AND_OPTIONS.md` (Options H–K are new), `CURRENT_STATUS.md`.

**Status: approved 2026-10-15 with the changes discussed (a28 keeps the diversity constraint from step 0; no round-6 diffusion model; export draws fresh noise per call; tuning decided later). Code implemented, see `WALKTHROUGH.md`.**

Hardware: two RTX A5000 (24 GB). Time: Day 2–5 for experiments and the final model (2026-10-15 to 10-18), Day 6 (10-19) report only.

---

## 1. What round 5 decided

| Finding (round 5) | Consequence for round 6 |
|---|---|
| a24/a25/a26 all eligible (diversity 1.5–1.8, a20: 0.07) | The noise problem is solved; keep noise injection (init 0.01) and a diversity term |
| ~1 dB lost vs round-4 twins; only ~25% is sample variance, ~75% a worse average output (over-smoothed, painterly, text blurred; worst on deblurring and structured artifacts) | Undo what weakened the reconstruction signal: full composite loss **on every sample**, 16 images per step (§4) |
| Loss on the mean of two samples halves each sample's reconstruction gradient; round-5 hinge was inactive (spread 2.6 > target 2.0) | Variance is bought by a constraint that is actually active, on a few extra draws, not by relaxing the loss |
| Wavelet discriminator: +0.23 dB `unseen_v2` in two independent pairs (a21−a20, a25−a24), neutral on v1 | Keep it in every round-6 model |
| Test-time noise ×0.5: ≤ 0.05 dB PSNR change, slightly better DISTS | Choose the noise scale per final model at evaluation, no training |
| a26 (diffusion + GAN) worst of the round; more steps always worse; diffusion has been ~0.3 dB behind the GAN under every shared recipe (a22 vs a20 −0.38, a26 vs a24 −0.27) | Diffusion closed. No round-6 diffusion + GAN: as a one-step model it is a GAN whose input carries extra noise, so the round-6 fixes would lift it as much as the GAN and the gap would stay |
| a25 = best eligible model (30.93 / 0.8372 / 0.1674) | Train it 200k now as the compliant fallback |

Decisions that stay: strictly generative (noise input, diversity ≥ 0.5, adversarial training); ~5 GFLOPs, one pass; from scratch; data v2; composite loss.

---

## 2. Full training of the best model now, or improvements first?

**Both, in parallel.**

- **GPU 0: a25 at 200k steps, starting today.** It is the best compliant model we have. For regression, 60k → 200k was worth +1.08 dB, so a25_full should land near ~31.9–32.0 dB. That is a submittable result whatever happens in round 6. It needs only a config (no new code).
- **GPU 1: round-6 screening (a27, then a28), 60k steps each.** If one of them beats a25 by ≥ 0.4 dB at 60k (same steps, like for like), its 200k run starts on GPU 0 when a25_full finishes (Day 3).
- Not doing: long runs of a24 or a26 (a25 beats them), a third screening round (no time).

---

## 3. Data (unchanged)

- Clean images: ten sources, natural 70% / medical 20% / low-light, 256×256 crops, flips/rotations, 4% grey.
- Corruptions: pipeline v2 (22 families, 0–4 per image, severity Beta(1.0, 1.8), random order).
- Testing: benchmark v1 (396 images, 33 cases), benchmark v2 (96 images, unseen families), diversity over 8 seeds, noise scales ×0.5 / ×1 / ×1.5.
- Batch: **16 different images per step again** (a27, a28), plus a second noise draw for 4 of them (20 generator outputs).
- Later, final model only (Day 4, optional): corruption list v3 as a fine-tune.

---

## 4. The fix shared by a27 and a28: "full loss + diversity constraint"

Why round 5 lost 1 dB: each sample received only ~0.7× a20's reconstruction gradient (½ from the loss on the mean + 0.2 per sample). It saw half as many images per step, and the adversarial term was 2× stronger relative to reconstruction on a discriminator near chance (loss 1.7–1.9 of 2.0). The result was a softer average output.

Change (training only, 0 inference FLOPs), after rcGAN (Bendel et al., NeurIPS 2023: per-sample penalty + variance reward with an adaptive weight):

1. **Composite loss on every generated sample at full weight** (as a20): Charbonnier 1.0 + FFT 0.05 + SSIM 0.2 + DISTS 0.1 from 50%.
2. **Diversity hinge as an active constraint:** for 4 of the 16 images draw a second noise; penalty `w · max(0, 1.5/255 − mean|x1 − x2|)`, **w = 4.0**. Target 1.5 grey levels of |x1 − x2| ≈ 1.3 std (above the 0.5 eligibility bar with margin; ~0.15 dB variance cost). The weight is set high because the full per-sample loss now pushes variance down, so the hinge must win. Monitor `train_spread255`: it should sit at ~1.5.
3. **Adversarial weight back to 0.005** (a20/a21), per sample, 5k ramp: in round 4 it cost almost nothing, and at 0.01 it added noise rather than texture.
4. Noise injection after every NAF block, strength init 0.01 (as a24/a25).

---

## 5. Round-6 models

All: NAF U-Net generator width 24, enc [2,2,2,4], middle 3, dec [1,1,1,2], residual output, `NoiseInjection` after every block. AdamW 4e-4 cosine → 1e-6, 1k warm-up, EMA 0.999, AMP, grad-clip 1.0, from scratch, data v2.

### a25_full: a25 trained 200k steps (fallback, starts now)

| Item | Setting |
|---|---|
| Model | exactly a25 (`architectures/a25_gan_div_wave.md`): generator + wavelet-domain conditional U-Net discriminator (Haar LH/HL/HH of candidate and corrupted image, 18 ch, 128 px) |
| Batch | 8 images × 2 draws |
| Losses | composite on mean of 2 samples + 0.2 Charbonnier per sample + diversity hinge (2/255, w 1.0) + wavelet L1 0.05 + hinge adversarial 0.01 + R1 |
| Frequency | wavelet adversarial + wavelet fidelity; FFT term in the composite loss |
| Schedule | **200k steps** (80 epochs × 2.5k), same cosine, DISTS from 100k |
| FLOPs | 4.98 GFLOPs, 5.52 M parameters |
| Expected | ~31.8–32.0 dB, DISTS ~0.14–0.15, diversity ~1.5 |

### a27_gan_wave_varctl: full per-sample loss + active diversity constraint (main candidate)

| Item | Setting |
|---|---|
| Generator | a25's (noise init 0.01) |
| Discriminator | a25's wavelet-domain conditional U-Net (spectral norm), training only |
| Batch | 16 images; 4 of them get a second draw → 20 generator outputs |
| Losses (G) | **composite on every output** (full weight) + wavelet L1 0.05 per output + diversity hinge on the 4 pairs (target 1.5/255, w 4.0) + hinge adversarial **0.005** per output (5k ramp) |
| Losses (D) | hinge + R1 (γ 1, every 16 steps), real = clean, fake = all 20 outputs |
| Frequency | wavelet-domain discriminator + wavelet fidelity (kept for the +0.23 dB unseen gain); FFT term in the composite loss |
| FLOPs | 4.98 GFLOPs, 5.52 M parameters |
| Question | does the full per-sample loss recover a21's quality (31.81) while keeping diversity ≥ 1? |
| Expected | 31.5–31.7 dB at 60k, diversity ~1.2–1.4; `unseen_v2` ≥ 30.5 |

### a28_gan_wave_2stage: a27 with the adversarial term delayed

| Item | Setting |
|---|---|
| Model, batch, losses | as a27 |
| Phase 1 (0–50% of steps) | composite + wavelet L1 on every output **and the diversity constraint (active from step 0)**; noise injection in use throughout; **no adversarial term, no discriminator training** |
| Phase 2 (50–100%) | adversarial term (0.005, ramped over 5k steps) and discriminator training switched on; everything else unchanged. DISTS also starts at 50% (as in every model) |
| Frequency | as a27 |
| FLOPs | 4.98 GFLOPs |
| Question | does a reconstruction warm-up (SRGAN / ESRGAN / Real-ESRGAN schedule) give a sharper model than adversarial training from step 0? |
| Is it generative? | Yes. It samples (noise input, diversity enforced) for the whole of training, and the adversarial, distribution-matching objective shapes the final 50%. Two-stage training is the standard recipe of GAN restorers (SRGAN, CVPR 2017; ESRGAN, ECCVW 2018; Real-ESRGAN, ICCVW 2021), and those have no noise input at all. Report wording: "conditional GAN with noise injection; reconstruction warm-up with a diversity constraint, then adversarial training" |
| Use rule | used if it beats a27 by ≥ 0.2 dB at 60k, with diversity ≥ 0.5 |
| Expected | 31.6–31.8 dB at 60k |

### Final run (Day 3–4): winner at 200k

The winner of {a27, a28} by §6 at 200k steps, with the a25_full schedule. If neither qualifies, a25_full is the final model and GPU 0 instead runs the width-16 sibling or the v3 fine-tune.

### Not in round 6, and why

- **Diffusion + GAN with the round-6 fixes:** the gap to the GAN is structural (§1); not run.

- LDL / focal-frequency / relativistic losses: the over-smoothing has a clear cause (weakened per-sample loss); adding new losses at the same time would hide whether the fix worked. Kept for the report as related work.
- Diffusion, distillation, flows (Options B–F, K): closed or out of time (`PROJECT_REVIEW_AND_OPTIONS.md` status table).
- RGB discriminator: the wavelet one matched it on v1 and won on unseen families twice.

---

## 6. How the winner is chosen

1. **Eligible:** diversity ≥ 0.5 at the chosen noise scale; FLOPs ≤ 5.1.
2. Round-6 screening: a27 or a28 goes to 200k only if its 60k PSNR is ≥ a25 + 0.4 dB (31.33) with DISTS no worse than a25 + 0.005. Otherwise a25_full is final.
3. Final choice among 200k models: rank PSNR, SSIM, DISTS separately (FLOPs equal); tie → better `unseen_v2`.
4. Reject a model whose medical or low-light loss vs a15 is > 0.5 dB beyond its overall loss.
5. Noise scale: the one in {0.5, 1.0} with the better average rank, keeping diversity ≥ 0.5.

---

## 7. Schedule, time and memory (two 24 GB GPUs)

Times from round 5 (measured): a25 5.5 h for 60k incl. evaluation (~765 s per 2.5k steps); a24 6.8 h. a27/a28 make 20 instead of 16 generator outputs per step: ~1.25× a25. Memory was not logged in the round-5 run; the estimates come from round 4/5 settings and are checked by the dry run (the step lines print `gpu_mem`).

| Run | Steps | Time | Peak GPU memory |
|---|---|---|---|
| a25_full | 200k | ~17 h + 0.5 h eval | ~12–14 GB |
| a27 | 60k | ~6.5 h + 0.5 h eval | ~14–17 GB |
| a28 | 60k | ~6 h + 0.5 h eval (stage 1 has no discriminator) | ~14–17 GB |
| winner (a27/a28) full | 200k | ~21 h + 0.5 h eval | ~14–17 GB |

| Day | GPU 0 | GPU 1 | Other |
|---|---|---|---|
| 2 (10-15) | **a25_full starts** as soon as its config exists (afternoon) | code a27/a28 (~2 h) + dry run, then **a27** (evening → ~03:00) | approve plan |
| 3 (10-16) | a25_full finishes (~07:00–09:00) → evaluate | **a28** (~03:00 → ~10:00) | **decision at ~10:00 by §6**; winner 200k starts on GPU 0 by noon (→ Day 4 ~10:00) |
| 4 (10-17) | winner 200k done (~10:00) → evaluate, noise-scale choice | free from Day 3: width-16 sibling (2.3 GFLOPs, 60k) or v3 fine-tune of the final model | pick final |
| 5 (10-18) | – | – | export (sampler inside the call, uint8), fvcore + ptflops, CPU test from clean checkout, diversity proof |
| 6 (10-19) | – | – | report only |

If the decision on Day 3 is "a25_full is final", GPU 0 is free from Day 3 morning: use it for the v3 fine-tune of a25_full (~12 h), which targets the unseen-family weakness.

---

## 8. Tuning of the final model

Decided after the final model is chosen (Day 4). Candidates, all keeping diversity ≥ 0.5: test-time noise scale; for a28, interpolation between the phase-1 and final weights (ESRGAN); short low-learning-rate fine-tunes (diversity target, adversarial weight); v3 corruption fine-tune.

## 9. Code (implemented 2026-10-15)

- `gan.py`: pair mode (`stochastic.pair_images`: full composite loss on every output, second draw for k images, diversity hinge on the pairs) and `gan.adv_start_frac` (adversarial term and discriminator from that fraction of training). Round-4/5 configs behave as before.
- `gennet.py`, `resdiff.py`: `fresh` noise mode and `set_fresh_noise()`.
- `export_inference.py`: the submission draws **fresh noise on every call** (`denoise_image(img)`); `denoise_image(img, seed=k)` gives a reproducible sample. The export check uses seed 0.
- New: `a25_full`, `a27_gan_wave_varctl`, `a28_gan_wave_2stage` (configs, models, registry).
