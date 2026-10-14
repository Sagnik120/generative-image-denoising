# Plan: round 5, make the generative model really sample, then train it long

Written 2026-10-14, after round 4 (a20–a23). Replaces the earlier 2026-10-14 plan (round 4). Background: `PROJECT_REVIEW_AND_OPTIONS.md` (options and sources), `COMPARISON.md` §9 (round-4 numbers), `CURRENT_STATUS.md`.

**Status: approved 2026-10-14; code for a24–a26 implemented (`WALKTHROUGH.md`). Hardware: two 24 GB GPUs, so the schedule below is the two-GPU version.**

Time available: 6 days. Days 1–5 for experiments and the final model, day 6 for the report only.

---

## 1. What round 4 decided

| Finding (round 4) | Consequence for round 5 |
|---|---|
| a20/a21 GANs: diversity 0.07 / 0.06, learned noise strength ≈ 0.002. The noise is switched off | The loss must stop punishing variance and must reward it (§4) |
| a20 costs only −0.15 dB vs a15 and trains stably; no damage on medical / low-light | Keep the a20 GAN machinery (U-Net SN discriminator, hinge, R1, ramp) as the base |
| a21 wavelet: −0.08 dB v1 (noise level), +0.24 dB `unseen_v2`, but tested with an adversarial term that had no influence | Re-test the wavelet discriminator once, inside a GAN whose adversarial term matters (a25) |
| a22 1-step diffusion: diversity 0.50, −0.52 dB, +0.012 DISTS; 2 and 4 steps worse | Diffusion only as diffusion + GAN, one step, training focused on the sampled level (a26) |
| a23 2-step w16: −2.1 dB, worst DISTS | No multi-step models at ≤ 5 GFLOPs. Closed |
| a19 vs a15: 200k steps = +1.08 dB | The final model must get the long schedule (§7, Day 2–4) |

Decisions that stay: strictly generative; ~5 GFLOPs, one pass, never above ~5.1; train from scratch (no a19 weights); data v2 unchanged during screening; composite loss kept.

---

## 2. Run the best round-4 model long now, or go straight to improved models?

**Straight to improved models.** With two GPUs a separate safety-net long run does not fit, so the safety net is what already exists.

- a20 is the best round-4 model on quality, but it is **not generative** (diversity 0.07). Training it for 200k steps would produce a better regression network, which we cannot submit. So no long run of a20 or a21.
- **Safety net:** the a22 checkpoint (60k, a genuine diffusion objective, diversity 0.50, raised by the test-time noise scale if needed) plus the best eligible round-5 checkpoint at 60k. Either is submittable if the long run fails.
- Both GPUs go to the round-5 screening, then one GPU to the long run of the winner as soon as the a24/a26 decision can be made.

---

## 3. Data (unchanged)

- **Clean images**: ten sources, natural 70% / medical 20% / low-light, 256×256 crops, flips/rotations, 4% grey.
- **Corruptions**: pipeline v2 (22 families, 0–4 per image, severity Beta(1.0, 1.8), random order, 65% of picks from the nine listed families).
- **Testing**: benchmark v1 (396 images, 33 cases), benchmark v2 (96 images, unseen families), diversity with 8 seeds.
- **Batch**: 16 generator outputs per step as **8 images × 2 noise draws** (needed for §4). Same compute per step as round 4; 8 instead of 16 different images per step.
- **Later, final model only** (Day 4): corruption list v3 as a fine-tune, if time allows.

---

## 4. The fix shared by all round-5 models: the "stochastic recipe"

Why the noise died: for a random output, expected error = (error of the average output)² + variance. Every term of the composite loss compares one output with one target, so variance is pure penalty. Three changes, all training-only (0 inference FLOPs):

1. **Reconstruction loss on the mean of two samples.** For each image draw two noises z1, z2, produce x̂1, x̂2, apply the composite loss to (x̂1 + x̂2)/2. This halves the variance penalty and keeps the average output, which is what PSNR/SSIM reward, close to the truth. A small per-sample Charbonnier (weight 0.2) keeps each single sample anchored. Idea: "averaging-consistency" in `PROJECT_REVIEW_AND_OPTIONS.md` §7; Whang et al., CVPR 2022, show the average of samples recovers PSNR.
2. **Diversity term with a target.** Hinge penalty `max(0, d_target − mean|x̂1 − x̂2|)`, d_target = 2 grey levels (2/255), weight 1.0. It pushes the two samples apart until they differ by ~2 grey levels and then stops, so it cannot run away and its PSNR price is bounded (~0.1–0.2 dB by the variance arithmetic). Source: diversity-sensitive / mode-seeking conditional GANs (Yang et al., "DSGAN", ICLR 2019; Mao et al., "Mode Seeking GANs", CVPR 2019), here with an explicit target instead of an unbounded ratio.
3. **Adversarial term that matters.** Weight 0.01 (2× round 4), ramped over the first 5k steps, applied to each sample. The discriminator is what makes the variance turn into realistic detail rather than noise. Noise-strength parameters start at 0.01 instead of 0.

Eligibility (unchanged): diversity ≥ 0.5 grey levels, FLOPs ≤ 5.1 GFLOPs. Aim: 1–2 grey levels.

Training cost: about the same per step as a20 (two half-batches instead of one full batch). 60k steps ≈ 9 h for a GAN, ≈ 6 h for diffusion.

---

## 5. Round-5 models

All: NAF U-Net width 24, enc [2,2,2,4], middle 3, dec [1,1,1,2], residual output, AdamW 4e-4 cosine → 1e-6, 1k warm-up, EMA 0.999, AMP, grad-clip 1.0, 60k steps, from scratch, data v2.

Composite loss (C) everywhere: Charbonnier 1.0 + FFT 0.05 + (1 − SSIM) 0.2 + DISTS 0.1 from 50% of training.

**Test-time noise scale (all models, no training).** Each model is evaluated with its noise multiplied by s ∈ {0.5, 1.0, 1.5}. This gives a curve of PSNR vs diversity from one checkpoint; we submit the s with the best quality that keeps diversity ≥ 0.5.

### a24_gan_div: GAN + stochastic recipe (main candidate)

| Item | Setting |
|---|---|
| Family | conditional GAN, one pass |
| Generator | a20 `GenNet`: NAF U-Net + StyleGAN noise injection after every NAF block (strength init 0.01) |
| Discriminator | a20's U-Net with spectral norm, conditional (sees corrupted + candidate, RGB, 6 channels), training only |
| Losses (G) | C on mean of 2 samples + 0.2 Charbonnier per sample + diversity hinge (target 2/255, weight 1.0) + hinge adversarial 0.01 per sample (5k ramp) |
| Losses (D) | hinge + R1 (γ = 1, every 16 steps) |
| Frequency | FFT magnitude term inside C (as before); discriminator in RGB |
| FLOPs | 4.98 GFLOPs, 5.52 M parameters |
| Expected | diversity 1–2; PSNR ≈ 31.6–31.8 (−0.1 to −0.3 vs a20); DISTS ≤ a20 |

### a25_gan_div_wave: a24 + wavelet discriminator (frequency variant)

| Item | Setting |
|---|---|
| Difference from a24 | discriminator input = Haar detail bands (LH, HL, HH) of candidate and corrupted image (18 ch, 128 px), as a21; + wavelet L1 (Haar bands, weight 0.05) |
| Frequency | wavelet-domain adversarial + wavelet fidelity (Korkmaz et al., CVPR 2024); FFT in C |
| FLOPs | 4.98 (training-only change) |
| Question it answers | does the wavelet discriminator help when the adversarial term has real influence? a25 − a24 = its value. Watch colour/brightness drift (it cannot see the low band) and `unseen_v2` |

### a26_resdiff_gan: one-step diffusion + GAN

| Item | Setting |
|---|---|
| Family | residual diffusion (ResShift forward process) with an adversarial loss on its sample, after Denoising Diffusion GANs (Xiao et al., ICLR 2022) |
| Network | a22 `ResDiffNet` (6-channel input `[y, x_t]`, time embedding by FiLM), width 24 |
| Forward process | T = 4, `x_t = x0 + η_t (y − x0) + κ √η_t ε`, κ = 0.1, η = (t/4)² |
| Level sampling | 75% at the last level (the only one used by the 1-step sampler; a22 used 50%), 25% spread over all levels |
| Sampler | 1 step from `y + κ ε`, noise drawn inside the model call |
| Losses (G) | stochastic recipe on the clean-image estimate: C on mean of 2 samples (two ε draws) + 0.2 Charbonnier per sample + diversity hinge + hinge adversarial 0.01 (same conditional RGB discriminator as a24) |
| Frequency | FFT in C |
| FLOPs | 5.02 GFLOPs, 5.68 M parameters |
| Expected | diversity ≥ 1 (the noise is in the input and is now rewarded); PSNR between a22 and a24. It is the most clearly "generative" story (diffusion + GAN) |

### Dropped for two GPUs

- **a22_full** (200k safety-net run): the existing a22 60k checkpoint is the safety net instead.
- **a27** (adversarial weight 0.02): run only if a GPU is idle on Day 3 and the winner's diversity is too low.

### What is deliberately not in round 5

- Multi-step sampling, distillation (Option C): closed by a22/a23; no time for a teacher–student pair.
- Wavelet- or latent-domain networks (Option E): not needed with one step.
- LDL, focal frequency, relativistic loss: held back, used in Day 4 tuning only if the winner shows the matching weakness (LDL for visible artifacts, focal frequency for weak texture).
- VAE latent noise (Option G): only if a24 still collapses despite the diversity term.

---

## 6. How the winner is chosen

1. **Eligible**: diversity ≥ 0.5 (at some noise scale s), FLOPs ≤ 5.1.
2. Among eligible: rank PSNR, SSIM, DISTS separately (FLOPs are equal); best average rank wins. Tie → better `unseen_v2`.
3. Reject if medical or low-light is more than 0.5 dB below the model's own overall loss vs a15 (adversarial artifacts).
4. Read the round-5 results against a20 (31.89 / 0.8554 / 0.1534): the price of real sampling = a24 − a20.

---

## 7. Schedule (5 working days + 1 for the report, two 24 GB GPUs)

GPU-hours from round 4 (a20 1300 s / 2.5k steps, a22 870 s): a24 ≈ 8–9 h, a25 ≈ 7 h, a26 ≈ 7–8 h for 60k steps; a 200k GAN ≈ 29 h, a 200k a26 ≈ 25 h.

| Day | GPU 0 | GPU 1 | Other work |
|---|---|---|---|
| 1 | dry run (15 min), then a24 (60k, ~9 h, overnight) | a26 (60k, ~8 h), then a25 (~7 h) | commit, pull, dry run |
| 2 | a24 done (morning) → evaluate; **choose a24 vs a26 by §6**; winner's 200k config (small code step); **start long run by midday** | a25 finishes (~midday) | if a25 clearly beats a24, its two changes are training-only: switch the long run to a25 only if it is still in its first hours |
| 3 | long run | free: width-16 sibling of the winner (60k, ~5 h) or a27 if diversity is low; clean-input check | report skeleton |
| 4 | long run done (afternoon / evening) → evaluate (v1, v2, diversity, noise-scale sweep) | optional v3 corruption fine-tune of the winner (overnight) | pick final configuration |
| 5 | | | final choice; export sampler (noise inside the call), uint8, fvcore + ptflops, CPU test, diversity proof |
| 6 | | | report only |

If round 5 fails completely (no eligible model), the submission is a22 (60k) with the noise scale that gives diversity ≥ 0.5, or a22 retrained 200k on Day 2–3 (~19 h).

---

## 8. What I need from you

- Approval of this plan, or the changes you want.
- Nothing open. Next: run the dry run and the screening (`WALKTHROUGH.md`).
