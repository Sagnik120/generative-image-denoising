# Model comparison (a05–a29_tuned)

> **Update 2026-10-20 (final):** round 8 done. **Final model: a29_tuned (Optuna trial 32) at noise ×0.75: 32.74 / 0.8740 / 0.1371 at 4.98 GFLOPs, diversity 0.77**: −0.37 dB from the regression a19, equal SSIM, best `unseen_v2` of all models. See §13.

> **Update 2026-10-18:** round 7 done. **a29_gan_decnoise is the final architecture: 32.27 / 0.8682 / 0.1360 at noise ×0.5, diversity 0.86, 4.98 GFLOPs**: −0.84 dB from the regression a19. It beats a25_full and a27_full on all three quality metrics at their best eligible noise scales. See §12.

> **Update 2026-10-16:** round 6 done. **a25_full (generative, 200k) is the best compliant model: 32.02 / 0.8576 / 0.1399 at noise ×0.5, diversity 1.04**: equal to the regression a15 on PSNR, better on DISTS, −1.1 dB vs a19. See §11.

> **Update 2026-10-15:** round 5 (a24–a26) gives the first eligible generative models (diversity 1.5–1.8), at ~1 dB below their round-4 counterparts; best eligible is a25 (30.93 / 0.8372 / 0.1674). See §10.

> **Update 2026-10-14:** round 4 (a20–a23, strictly generative models) is now benchmarked; see §9. None of them is yet a satisfactory generative submission: the two GANs ignore their noise, the 1-step diffusion model is borderline at −0.52 dB, the 2-step one is −2.1 dB. Sections 1–8 compare the regression models and remain valid as reference.

> **Update 2026-10-11:** the brief requires a *generative* model. a19 is a regression network and is **not a compliant submission**; it is now the upper reference and the initialisation for the generative model. See `CURRENT_STATUS.md` and `PLAN.md`. The measurements below remain valid.

Written 2026-10-10 from `results/comparison/*.csv` and each model's `final_evaluation_summary.json`.

All 15 models below were scored on the same fixed benchmark v1: 396 hold-out images (198 natural, 99 low-light, 99 medical), 33 degradation cases, mean input PSNR 24.49 dB. Round 1 (a01–a04) used a different validation set and is left out. Every run is single-seed; treat differences under ~0.05 dB as noise.

## 1. Answer

**a19_final is the best model, by a wide margin** (best regression model; not submittable, see the note above).

- 33.11 dB / 0.8743 SSIM / 0.1293 DISTS at 4.98 GFLOPs.
- +1.08 dB PSNR, +0.015 SSIM, −0.024 DISTS over the next best model (a15) at identical FLOPs.
- Best PSNR in all 33 cases; best PSNR, SSIM and DISTS in all five groups and all three domains; best on the unseen-family benchmark v2.
- It is architecturally the simplest model in the table: a plain NAF U-Net. It wins because of **what it was trained on (v2 degradations) and for how long (200k steps)**, not because of its structure.

## 2. Full table

Sorted by PSNR. "Train" = degradation pipeline / steps.

| Model | Idea | Train | PSNR | SSIM | DISTS | GFLOPs | Params (M) |
|---|---|---|---|---|---|---|---|
| **a19_final** | a15, long schedule | v2 / 200k | **33.112** | **0.8743** | **0.1293** | 4.978 | 5.52 |
| a15_naf_degv2 | NAF + v2 degradations | v2 / 60k | 32.034 | 0.8594 | 0.1537 | 4.978 | 5.52 |
| a16_naf_attn_degv2 | a15 + attention bottleneck | v2 / 60k | 31.989 | 0.8582 | 0.1550 | 4.853 | 4.19 |
| a05_nafnet_v2 | plain NAF U-Net (reference) | v1 / 60k | 31.924 | 0.8550 | 0.1542 | 4.978 | 5.52 |
| a09_wavelet_naf | NAF in Haar domain | v1 / 60k | 31.891 | 0.8543 | 0.1545 | 5.018 | 6.44 |
| a07_naf_attn_hybrid | NAF + attention bottleneck | v1 / 60k | 31.854 | 0.8558 | 0.1563 | 4.928 | 4.02 |
| a11_freq_spatial | NAF + FFT branch | v1 / 60k | 31.838 | 0.8552 | 0.1572 | 4.884 | 5.27 |
| a08_degradation_aware | a07 + FiLM degradation embedding | v1 / 60k | 31.835 | 0.8530 | 0.1612 | 4.982 | 4.23 |
| a17_naf_dists0 | DISTS loss from step 0 | v1 / 60k | 31.833 | 0.8530 | 0.1517 | 4.978 | 5.52 |
| a06_restormer_v2 | Restormer | v1 / 60k | 31.682 | 0.8520 | 0.1623 | 4.984 | 1.31 |
| a18_naf_dists0_strong | DISTS 0.2 + SSIM 0.3 from step 0 | v1 / 60k | 31.566 | 0.8476 | 0.1502 | 4.978 | 5.52 |
| a12_naf_w16 | a05 at width 16 | v1 / 60k | 31.437 | 0.8468 | 0.1682 | **2.278** | 2.47 |
| a14_wavelet_w28 | a09 at width 28 | v1 / 60k | 31.359 | 0.8455 | 0.1687 | 2.522 | 3.18 |
| a13_naf_w12 | a05 at width 12 | v1 / 60k | 30.871 | 0.8364 | 0.1856 | **1.318** | 1.40 |
| a10_resflow_fewstep | 2-step residual flow (generative) | v1 / 60k | 30.349 | 0.8238 | 0.1976 | 5.039 | 2.86 |

Internal rank table (`comparison_ranking.csv`, each axis ranked separately as the competition does): a19 avg rank 3.0 (1/1/1 on quality, tied on FLOPs with the other width-24 NAF models), then a15 and a16 at 4.25, a05 and a07 at 6.0. The FLOPs rank among our own models says little, since eleven of them sit at 4.85–5.04 GFLOPs; what matters is where 5 GFLOPs falls among other teams (§7).

## 3. What each change was worth

Every round-3 run changes one thing against a reference, so the effects can be read off directly.

| Change | Comparison | ΔPSNR | ΔSSIM | ΔDISTS | ΔGFLOPs | Kept? |
|---|---|---|---|---|---|---|
| **Training 60k → 200k steps** | a19 vs a15 | **+1.078** | **+0.0149** | **−0.0244** | 0 | yes |
| Degradation pipeline v1 → v2 | a15 vs a05 | +0.110 | +0.0044 | −0.0005 | 0 | yes |
| Global attention bottleneck (v2 data) | a16 vs a15 | −0.045 | −0.0012 | +0.0013 | −0.125 | no |
| Global attention bottleneck (v1 data) | a07 vs a05 | −0.070 | +0.0008 | +0.0021 | −0.050 | no |
| DISTS loss from step 0 | a17 vs a05 | −0.091 | −0.0020 | −0.0025 | 0 | no |
| DISTS 0.2 + SSIM 0.3 from step 0 | a18 vs a05 | −0.358 | −0.0074 | −0.0040 | 0 | no |
| Haar wavelet front end, 5 GFLOPs | a09 vs a05 | −0.033 | −0.0007 | +0.0003 | +0.040 | no |
| FFT branch | a11 vs a05 | −0.086 | +0.0002 | +0.0030 | −0.094 | no |
| Degradation embedding (FiLM) | a08 vs a07 | −0.019 | −0.0028 | +0.0049 | +0.054 | no |
| Restormer blocks instead of NAF | a06 vs a05 | −0.242 | −0.0030 | +0.0081 | +0.006 | no |
| Generative 2-step flow | a10 vs a05 | −1.575 | −0.0312 | +0.0434 | +0.061 | no |
| Width 24 → 16 | a12 vs a05 | −0.487 | −0.0082 | +0.0140 | −2.700 | fallback |
| Width 24 → 12 | a13 vs a05 | −1.053 | −0.0186 | +0.0314 | −3.660 | no |
| Wavelet at 2.5 GFLOPs | a14 vs a12 | −0.078 | −0.0013 | +0.0005 | +0.244 | no |

Reading it:

1. **Training length dominates everything.** +1.08 dB from one config line. The whole spread of seven architectures in round 2 was 0.09 dB among the top five. Nothing structural we tried is within a factor of ten of it.
2. **Data is second.** v2 degradations are worth +0.11 dB on v1 overall, but that average hides the real effect: +2 to +8 dB on the families added, −0.15 dB on the old ones at 60k steps (recovered by long training).
3. **Every architectural addition was neutral or negative** at equal FLOPs: attention, FFT, wavelet, conditioning, Restormer, flow.
4. **Loss weighting moves you along a trade-off curve, it does not shift the curve.** a17/a18 buy DISTS with PSNR and SSIM at a worsening rate.
5. **FLOPs can be cut, at a convex price:** 0.18 dB per GFLOP for the first 2.7 GFLOPs, 0.59 dB per GFLOP for the next one.

## 4. By group and domain (PSNR, dB)

| Model | single | extra | pair | triple | unseen (v1) | natural | low-light | medical |
|---|---|---|---|---|---|---|---|---|
| **a19** | **36.28** | **37.09** | **31.18** | **29.96** | **32.75** | **29.50** | **37.99** | **35.45** |
| a15 | 34.89 | 34.52 | 30.53 | 29.36 | 31.61 | 28.57 | 36.94 | 34.05 |
| a16 | 34.84 | 34.49 | 30.46 | 29.30 | 31.64 | 28.54 | 36.89 | 33.99 |
| a05 | 35.05 | 35.12 | 30.68 | 29.55 | 29.19 | 28.47 | 36.82 | 33.94 |
| a09 | 34.99 | 35.17 | 30.64 | 29.58 | 29.10 | 28.40 | 36.76 | 34.01 |
| a07 | 34.84 | 34.74 | 30.63 | 29.55 | 29.52 | 28.42 | 36.74 | 33.84 |
| a17 | 35.09 | 34.85 | 30.58 | 29.46 | 28.98 | 28.39 | 36.61 | 33.94 |
| a18 | 34.76 | 34.36 | 30.40 | 29.26 | 28.68 | 28.17 | 36.30 | 33.63 |
| a12 | 34.50 | 33.94 | 30.25 | 29.26 | 28.94 | 27.99 | 36.37 | 33.40 |
| a14 | 34.30 | 33.85 | 30.22 | 29.27 | 28.89 | 27.89 | 36.37 | 33.28 |
| a13 | 33.57 | 32.87 | 29.83 | 28.88 | 28.91 | 27.44 | 35.98 | 32.62 |
| a10 | 32.92 | 32.38 | 29.31 | 28.51 | 28.45 | 27.06 | 35.36 | 31.92 |

DISTS by domain for a19 / a15 / a05: natural 0.134 / 0.157 / 0.157, low-light 0.152 / 0.177 / 0.180, medical 0.098 / 0.123 / 0.123.

Points:

- For the v2-trained models (a15, a16, a19) the v1 "unseen" column is no longer unseen: `pixelate` and `periodic_noise` are in their training set. That is why it jumps 2.4 dB. Use §5 for honest generalisation numbers.
- a19's gain over a15 is largest on single (+1.39) and extra (+2.56) and smallest on pair (+0.65) and triple (+0.61). Long training sharpens the inversion of single, deterministic corruptions; stacked blur + noise + JPEG is information-limited and moves little.
- The domain order never changes across models: low-light > medical > natural. Natural images carry the most texture and have the lowest PSNR in every model. Low-light has high PSNR (dark images have small absolute errors) but the worst DISTS.
- Ordering between models is the same in every domain. No model is a "specialist"; there is no case for a per-domain ensemble or router.

## 5. Generalisation: benchmark v2

96 images. `unseen_v2` (film grain, glass blur, line dropout, two mixes) is in no model's training set. `plan` (sensor noise, ringing, double JPEG) is trained on only by v2-pipeline models.

| Model | unseen_v2 PSNR | SSIM | DISTS | plan PSNR | SSIM | DISTS |
|---|---|---|---|---|---|---|
| **a19** | **31.88** | **0.8290** | **0.1736** | **37.38** | **0.9487** | **0.0708** |
| a15 | 30.99 | 0.8235 | 0.1865 | 36.21 | 0.9429 | 0.0823 |
| a17 | 30.91 | 0.8094 | 0.1951 | 33.22 | 0.9134 | 0.1062 |
| a05 | 30.85 | 0.8112 | 0.1985 | 33.18 | 0.9118 | 0.1099 |
| a18 | 30.79 | 0.8119 | 0.1959 | 33.05 | 0.9102 | 0.1086 |
| a12 | 30.39 | 0.8143 | 0.2039 | 33.02 | 0.9098 | 0.1149 |
| a16 | 30.18 | 0.8119 | 0.1989 | 36.43 | 0.9433 | 0.0826 |
| a14 | 30.04 | 0.8054 | 0.2074 | 33.06 | 0.9100 | 0.1148 |
| a13 | 29.28 | 0.7917 | 0.2254 | 32.97 | 0.9081 | 0.1205 |

Per unseen case, PSNR (input → a15 → a19):

| Case | Input | a15 | a19 | a19 gain over input |
|---|---|---|---|---|
| `line_dropout` | 19.69 | 35.40 | 38.97 | +19.28 |
| `line_dropout+jpeg` | 19.21 | 32.95 | 34.04 | +14.84 |
| `film_grain` | 25.09 | 28.25 | 28.44 | +3.35 |
| `glass_blur+film_grain` | 23.74 | 26.48 | 26.46 | +2.72 |
| `glass_blur` | 31.40 | 31.87 | 31.48 | **+0.09** |

Points:

- **Trained-on families gain 3–8 dB; never-seen families gain 0.1 dB from the same data change** (a15 vs a05: `plan` +3.03, `unseen_v2` +0.14). Models restore what they have been shown. The practical consequence is to keep adding plausible families to training.
- a19's `unseen_v2` lead over a15 (+0.89 dB) is almost all `line_dropout`, which resembles trained families (`stripe_noise`, `small_artifacts`, `salt_pepper`). On `glass_blur`, which resembles nothing in training, a19 is *worse* than a15 by 0.39 dB and barely better than doing nothing.
- The attention model a16 is the worst width-24 model on `unseen_v2`. The round-2 argument for attention (better on unseen) reversed once the test families changed, so it was a coincidence of one family (`periodic_noise`).

## 6. Head-to-head: why a19 and not the others

| Rival | What it has that a19 lacks | Why a19 is still preferred |
|---|---|---|
| a15 | nothing (same config, shorter) | a19 is strictly better; a15's only edge is `glass_blur` +0.39 dB |
| a16 / a07 | global attention | no gain once periodic patterns are trained on; −0.82 dB on `unseen_v2` |
| a17 / a18 | DISTS loss from the start, heavier | at 60k: −0.003/−0.004 DISTS for −0.09/−0.36 dB. a19's DISTS (0.129) is already far below both through training length |
| a09 / a14 | half-resolution wavelet processing | ties at 5 GFLOPs, loses at 2.5; weak on impulses |
| a08 | degradation embedding | won `salt_pepper` at 60k by 0.8 dB; a19 is now 3.4 dB above it there |
| a11 | FFT branch | no blur advantage appeared; FLOPs counting ambiguous |
| a12 / a13 | 54% / 74% fewer FLOPs | 0.49 / 1.05 dB lower at equal training; the one real alternative, see §7 |
| a10 | it is explicitly generative | 1.6 dB lower at 60k and worse DISTS; only relevant if the organisers insist on a GAN/diffusion/flow model |

## 7. The one open comparison: 5 GFLOPs vs 2.3 GFLOPs

a19 is unbeaten on quality, but FLOPs is a quarter of the score and our model sits at 4.98 GFLOPs. From the size study (60k steps, v1 data):

| Model | GFLOPs | PSNR | SSIM | DISTS |
|---|---|---|---|---|
| a05 | 4.98 | 31.92 | 0.8550 | 0.1542 |
| a12 | 2.28 | 31.44 | 0.8468 | 0.1682 |
| a13 | 1.32 | 30.87 | 0.8364 | 0.1856 |

A width-16 model trained like a19 (v2, 200k) has not been run. If the gap carries over, it would land near 32.6 dB / 0.866 / 0.143: about a19's quality minus 0.5 dB, at 46% of the FLOPs, and still above every 60k model in the table. Whether that is a better *competition* entry depends on the other teams' spread, which we cannot see:

- It gains FLOPs rank only for each team that sits between 2.3 and 5 GFLOPs.
- It loses PSNR, SSIM and DISTS rank for each team that sits within 0.5 dB / 0.008 SSIM / 0.014 DISTS below a19.

Quality counts on three axes, and a class competition usually has quality scores bunched more tightly than FLOPs, so my judgement is that a19 at 5 GFLOPs is the safer entry. This is a judgement, not a measurement; the plan (`PLAN.md`) includes one run to have the width-16 model ready in case the allowed number of submissions or a leaderboard makes the choice clearer.

## 8. Limits of this comparison

- **Single seed everywhere.** Gaps below ~0.05 dB (a05 vs a09, a07 vs a11 vs a08 vs a17) are not real orderings.
- **The benchmark is generated by our own degradation code** with the same severity ranges as training. It measures in-distribution quality. The private set uses someone else's implementations and severities; expect lower absolute numbers there, and expect the ranking of v2-trained vs v1-trained models to hold but the margins not to.
- **Checkpoint selection used benchmark v1**, so v1 numbers are very slightly optimistic. Benchmark v2 was not used for selection.
- **Unequal training:** a19 has 3.3× the steps of everything else. The fair statement is "a15's configuration is best at 60k, and it scales well with training", not "a19's architecture is better".
- 12 images per case. Single-case differences under ~0.3 dB are within image-sampling noise.

## 9. Round 4: strictly generative models (a20–a23)

All trained from scratch, v2 data, 60k steps, so **a15 is the like-for-like regression reference**; a19 (200k) is the upper reference. Diversity = mean per-pixel std over 8 noise seeds, grey levels of 255 (`results/<model>/generative_eval.json`); eligibility ≥ 0.5. Source: `results/comparison_round4/*.csv`.

### 9.1 Overall

| Model | Family | PSNR | SSIM | DISTS | GFLOPs | Diversity | Δ PSNR vs a15 | unseen_v2 | Eligible |
|---|---|---|---|---|---|---|---|---|---|
| a19 (ref.) | regression, 200k | 33.112 | 0.8743 | 0.1293 | 4.98 | 0 | +1.08 | 31.88 | no (regression) |
| a15 (ref.) | regression, 60k | 32.034 | 0.8594 | 0.1537 | 4.98 | 0 | 0 | 30.99 | no (regression) |
| a20_gan | GAN | **31.886** | **0.8554** | **0.1534** | 4.98 | 0.07 | −0.15 | 30.48 | no (noise ignored) |
| a21_gan_wavelet | GAN + wavelet D/loss | 31.809 | 0.8551 | 0.1544 | 4.98 | 0.06 | −0.23 | **30.71** | no (noise ignored) |
| a22_resdiff | residual diffusion, 1 step | 31.511 | 0.8499 | 0.1654 | 5.02 | 0.50 | −0.52 | 29.72 | borderline |
| a23_resdiff_2step | residual diffusion, w16, 2 steps | 29.930 | 0.8115 | 0.2119 | 4.61 | 1.26 | −2.10 | 28.18 | yes, but worst |

Round-4 rank table (all 19 benchmarked models, each axis ranked separately): a20 is 4th overall (avg rank 6.5), behind a19, a15, a16; a21 9th; a22 17th; a23 18th.

### 9.2 By group (PSNR, dB)

| Group | a15 | a20 | a21 | a22 | a23 |
|---|---|---|---|---|---|
| single | 34.89 | 34.61 | 34.50 | 34.21 | 32.13 |
| extra | 34.52 | 34.71 | 34.78 | 33.60 | 32.06 |
| pair | 30.53 | 30.37 | 30.30 | 30.10 | 28.80 |
| triple | 29.36 | 29.21 | 29.17 | 29.10 | 28.10 |
| unseen | 31.61 | 31.53 | 31.37 | 31.11 | 29.06 |
| natural | 28.57 | 28.46 | 28.30 | 28.08 | 26.66 |
| lowlight | 36.94 | 36.75 | 36.71 | 36.52 | 35.04 |
| medical | 34.05 | 33.89 | 33.93 | 33.36 | 31.36 |
| unseen_v2 | 30.99 | 30.48 | 30.71 | 29.72 | 28.18 |
| plan (v2) | 36.21 | 35.98 | 35.86 | 36.39 | 33.77 |

No GAN artifacts on medical or low-light: both GANs lose ≤ 0.2 dB there, about their overall loss.

### 9.3 Per-case pattern

- **GAN vs a15:** losses spread thin (most cases −0.1 to −0.3 dB); largest on `chromatic_aberration` −0.60 and `jpeg` −0.58; gain on `small_artifacts` +0.80.
- **Diffusion (a22) vs a15:** losses concentrate on structured, information-preserving corruptions: `small_artifacts` −2.33, `salt_pepper` −1.30, `periodic_noise` −1.25, `chromatic_aberration` −1.09, `motion_blur` −1.04. Random-noise cases only −0.2 to −0.3. The sampling noise in `x_T` hides the fine structure these cases need.
- **a23:** the same cases at −5.4 to −6.3 dB: the capacity-limited cases of the size study (as for a10).

### 9.4 What each comparison isolates

| Comparison | Isolates | Result |
|---|---|---|
| a20 − a15 | weak adversarial term (0.005) + learned noise | −0.15 dB, equal DISTS, noise switched off |
| a21 − a20 | wavelet discriminator + wavelet L1 | −0.08 dB v1 (noise level), +0.24 dB `unseen_v2` |
| a22 − a15 | residual diffusion objective, 1 step | −0.52 dB, +0.012 DISTS; diversity 0.50 |
| a22 at 1 → 2 → 4 steps | more sampling, no retraining | −0.30, −0.83 dB; DISTS also worse |
| a23 − a22 | half width, 2 steps, same FLOPs | −1.58 dB; second step itself −0.17 dB |

### 9.5 Conclusions

1. **The composite loss kills randomness.** For a random output, expected per-image error = error of the mean + variance. Every reconstruction term therefore pays the model to remove variance; the GAN obeys by switching its noise off (strength 0.0017), the diffusion model by erasing its start noise (and paying PSNR for the effort). Being generative needs a loss that does not punish variance (reconstruction on the **mean of several samples**) and a term that rewards it (adversarial at a meaningful weight, diversity-sensitive term).
2. **One full-size pass is the only viable budget.** a23 and the a22 step sweep confirm a10: every extra step at ≤ 5 GFLOPs costs 0.3–2 dB.
3. **The GAN machinery is sound.** a20 trains stably and costs only 0.15 dB at 60k, so a GAN that keeps the a20 recipe and adds the variance fix has the best chance of staying close to regression quality.
4. **The wavelet discriminator is undecided**, not refuted: it was tested where the adversarial term had almost no influence.
5. **Diffusion with reconstruction loss alone is not worth its cost.** It needs an adversarial term to make its noise produce realistic detail (diffusion + GAN), and training focused on the level actually sampled.

Next experiments (round 5): `PLAN.md`.

## 10. Round 5: stochastic recipe (a24–a26)

Same data, backbone and 60k steps as round 4, from scratch; 8 images × 2 noise draws per step; composite loss on the mean of the two samples + 0.2 Charbonnier per sample + diversity hinge (2 grey levels) + adversarial 0.01. Source: `run_round5.log`, `results/a24..a26/`.

### 10.1 Overall

| Model | PSNR | SSIM | DISTS | GFLOPs | Diversity | unseen_v2 | Δ PSNR vs round-4 twin | Eligible |
|---|---|---|---|---|---|---|---|---|
| a20 (ref., round 4) | 31.886 | 0.8554 | 0.1534 | 4.98 | 0.07 | 30.48 | – | no |
| a24_gan_div | 30.907 | 0.8349 | **0.1666** | 4.98 | 1.69 | 29.68 | −0.98 (a20) | yes |
| **a25_gan_div_wave** | **30.934** | **0.8372** | 0.1674 | 4.98 | 1.54 | **29.91** | −0.88 (a21) | **yes, best** |
| a26_resdiff_gan | 30.637 | 0.8299 | 0.1775 | 5.02 | 1.79 | 28.46 | −0.87 (a22) | yes, rejected |

Ranks among all 22 benchmarked models (each axis separately, average rank): a25 18th, a24 19th, a26 21st. Every eligible model is now below every regression model trained on v2 data.

### 10.2 Noise-scale sweep (test-time multiplier, same checkpoint)

| Model | ×0.5 PSNR / DISTS / div. | ×1 | ×1.5 |
|---|---|---|---|
| a24 | 30.94 / 0.1647 / 1.20 | 30.91 / 0.1666 / 1.69 | 30.81 / 0.1747 / 1.96 |
| a25 | 30.92 / 0.1643 / 1.13 | 30.93 / 0.1674 / 1.54 | 30.89 / 0.1762 / 1.70 |
| a26 | 30.88 / 0.1764 / 0.97 | 30.64 / 0.1775 / 1.79 | 30.19 / 0.1862 / 2.60 |

For the GANs the noise level barely matters for PSNR (±0.05 dB) but more noise costs DISTS; ×0.5 is the best submission point (still ≥ 1.1 diversity). For a26 the noise is mostly harm.

### 10.3 Where the dB went (a24 vs a20)

- **Error budget:** RMS error 6.47 → 7.29 grey levels (MSE 41.8 → 53.2). Sample variance 1.69² ≈ 2.9, i.e. **~25%** of the increase. **~75% is a worse average output.**
- **What it looks like:** over-smoothed, painterly outputs: fine texture (scales, coral, water, grass) flattened, small text blurred. Not grain.
- **Which cases:** structured / exact-inverse cases and deblurring lose most (`small_artifacts` −3.90, `periodic_noise` −1.65, `salt_pepper` −1.57, `chromatic_aberration` −1.36, blurs −1.2); `jpeg` −0.43, `resize_blur` −0.16. Domains equal (≈ −1 dB each): no adversarial artifacts.
- **Cause (by elimination):**
  - not the test-time variance (noise sweep: ±0.05 dB);
  - not the feature-level noise injection (a26 has none and lost the same 0.87 dB);
  - shared by a24 and a26: per-sample reconstruction gradient cut to ~0.7× (loss on the mean of two), 8 instead of 16 images per step, adversarial weight 2× on a near-chance discriminator (D loss 1.73–1.86 of 2.0). The gap to a20 is already 0.87 dB at epoch 4, before the batch effect can matter much, so the loss reweighting is the main suspect.

### 10.4 Wavelet discriminator, second confirmation

| Pair | v1 PSNR | unseen_v2 |
|---|---|---|
| a21 − a20 | −0.08 | **+0.24** |
| a25 − a24 | +0.03 | **+0.23** |

Neutral on the training families, consistently +0.23 dB on never-seen families. Helps noise cases, hurts deblurring and periodic patterns (it sees only the detail bands).

### 10.5 Conclusions

1. **Eligibility is solved; quality is not.** All three models really sample (diversity 1.5–1.8 vs 0.07).
2. **Variance is cheap; the way it was bought was not.** ~0.25 dB for the variance, ~0.75 dB for weakening the per-sample loss and the batch. The fix is to keep the per-sample composite loss at full weight (a20) and enforce a small variance by a constraint on a few extra draws (round 6, `PLAN.md`).
3. **Keep the wavelet discriminator** for the final model (+0.23 dB unseen, twice).
4. **Diffusion is closed** at ≤ 5 GFLOPs (a22, a23, a26 all behind the GAN; more steps always worse).
5. **Training length is still untested for generative models**; for regression it was worth +1.08 dB (a15 → a19).

## 11. Round 6: long run and the full per-sample loss (a25_full, a27, a28)

Source: `run_round6.log`, `results/comparison_round6/`, `results/<model>/`.

### 11.1 Overall

| Model | Steps | PSNR ×1 / ×0.5 | SSIM (×1) | DISTS (×1) | Diversity ×1 / ×0.5 | unseen_v2 | Avg rank (of 25) |
|---|---|---|---|---|---|---|---|
| a19 (regression ref.) | 200k | 33.112 | 0.8743 | 0.1293 | 0 | 31.88 | 1st |
| **a25_full** | 200k | **32.005 / 32.024** | **0.8566** | **0.1410** | 1.34 / 1.04 | **31.30** | **3rd** |
| a15 (regression ref.) | 60k | 32.034 | 0.8594 | 0.1537 | 0 | 30.99 | 4th |
| a20 (round 4, noise ignored) | 60k | 31.886 | 0.8554 | 0.1534 | 0.07 | 30.48 | 6th |
| a27_gan_wave_varctl | 60k | 31.237 / 31.391 | 0.8489 | 0.1601 | 1.10 / 0.64 | 30.45 | 17th |
| a28_gan_wave_2stage | 60k | 31.154 / 31.451 | 0.8511 | 0.1582 | 1.28 / 0.74 | 30.17 | 15th |
| a25 (round 5) | 60k | 30.934 / 30.921 | 0.8372 | 0.1674 | 1.54 / 1.13 | 29.91 | 21st |

### 11.2 What each comparison isolates

| Comparison | Isolates | Result |
|---|---|---|
| a25_full − a25 | 200k vs 60k steps | +1.07 dB, +0.019 SSIM, −0.026 DISTS (regression: +1.08 dB) |
| a25_full − a19 | generative vs regression at 200k | −1.11 dB, −0.018 SSIM, +0.012 DISTS |
| a27 − a25 | full per-sample loss, 16 images, adv 0.005, target 1.5 | +0.30 dB (×1), +0.47 (×0.5), −0.007 DISTS |
| a28 − a27 | adversarial from 50% instead of 0 | −0.08 dB; low-light −0.48, `unseen_v2` −0.28 |
| a28 phase 1 (epoch 12) − a20 (epoch 12) | forced diversity, no adversarial vs adversarial, no diversity | **−0.70 dB** |
| a27 (epoch 12) − a28 (epoch 12) | adversarial term on vs off, same diversity | +0.02 dB (none) |

### 11.3 Where the generative models lack (a25_full vs a19; a27 vs a20)

1. **Structured / exact-inverse corruptions:** `small_artifacts` −4.9, `periodic_noise` −2.4, `salt_pepper` −2.1, `chromatic_aberration` −1.8, blurs −1.1 to −1.5 dB. The same cases lose −1 to −3.8 dB in every model whose noise enters the encoder path (a22, a24–a28) and not in a20 (noise switched off). The injected noise disturbs the features that locate and invert these precise patterns.
2. **Mild corruptions are over-processed:** `resize_blur` below the input (a25_full 33.78, a27 33.37 vs input 34.24); `jpeg` ≈ input (35.20 vs 35.14; a19 36.43). A fixed amount of variation everywhere costs most where the input is already nearly clean.
3. **Fine detail under heavy noise is smoothed** (small spots, lettering): shared with the regression models, slightly stronger.
4. **No adversarial artifacts:** domains lose evenly (natural −1.06, low-light −1.23, medical −1.09 vs a19); DISTS is the axis where the generative model is closest to a19.

### 11.4 Conclusions

1. **a25_full is a strong, compliant submission now** (3rd of 25 overall; generative; 2nd-best DISTS of all models).
2. **The round-6 recipe is better than round 5's** (+0.47 dB at 60k, best noise scale). Trained 200k it should land at about 32.3–32.5 dB.
3. **The remaining cost is where the noise enters, not the adversarial term:** forced diversity costs ~0.7 dB with or without an adversarial loss. Next step: put the noise where it can only add stochastic detail (decoder), and let the network choose per pixel how much (round 7, `PLAN.md`).
4. **The two-phase schedule (a28) brings nothing** and is dropped.

## 12. Round 7: final long runs on data v3 (a27_full, a29)

Source: `run_round7.log`, `results/comparison_round7/`, `results/<model>/`. Both 200k steps, data v3, same recipe; only the noise placement differs. Compared at each model's **best eligible noise scale** (diversity ≥ 0.6), which is how the submission will run.

### 12.1 Overall

| Model | Scale | PSNR | SSIM | DISTS | Diversity | unseen_v2 (×1) | v3 check (×1) | Training at the end |
|---|---|---|---|---|---|---|---|---|
| a19 (regression ref.) | – | 33.112 | 0.8743 | 0.1293 | 0 | 31.88 | – | stable |
| **a29_gan_decnoise** | **×0.5** | **32.269** | **0.8682** | **0.1360** | 0.86 | 30.91 | 33.45 / 0.112 | **stable** (spread 1.65) |
| a27_full | ×0.6 | 32.184 | 0.8659 | 0.1444 | 0.675 | **31.42** | 33.50 / 0.119 | **unstable** (spread 4.5, −0.8 dB) |
| a25_full (v2 data) | ×0.5 | 32.024 | 0.8576 | 0.1399 | 1.04 | 31.30 | – | stable |

At noise ×1 (the comparison tables' default), a27_full ranks 2nd of 27 overall and a29 6th: ×1 is a29's worst setting (+0.34 dB from ×0.5), and its 4.982 GFLOPs put it behind the 4.978 models in the FLOPs rank by 0.004 GFLOPs.

### 12.2 What each comparison isolates

| Comparison | Isolates | Result |
|---|---|---|
| a29 − a27_full (same data, recipe, steps) | noise only in the decoder + per-pixel gate | +0.09 dB, +0.002 SSIM, **−0.008 DISTS** at best scales; **stable vs unstable** training |
| a27_full − a25_full | round-6 recipe + data v3 (confounded) | +0.16 dB, +0.008 SSIM, +0.005 DISTS (best checkpoints) |
| a29 − a25_full | round-6 recipe + data v3 + gated decoder noise | **+0.25 dB, +0.011 SSIM, −0.004 DISTS** |
| a29 − a19 | generative vs regression, 200k | −0.84 dB, −0.006 SSIM, +0.007 DISTS (round 6: −1.09 / −0.017 / +0.011) |

### 12.3 Training stability (the decisive difference)

| Epoch | 41 | 49 | 59 | 69 | 80 |
|---|---|---|---|---|---|
| a27_full val PSNR / spread | 32.15 / 2.35 | 31.68 / 1.74 | 31.70 / 1.68 | 31.60 / 2.86 | 31.32 / **4.48** |
| a29 val PSNR / spread | 32.13 / 1.97 | 32.00 / 1.88 | 31.85 / 1.71 | 31.91 / 1.67 | 31.93 / 1.65 |

With noise in all 18 blocks and only a lower bound on diversity, a27_full drifted towards more and more randomness late in training (as its learning rate fell and the discriminator sharpened) and lost 0.83 dB from its peak. Its reported numbers come from an early checkpoint chosen on the reported benchmark. a29's gated decoder noise kept the spread at the target and its last checkpoint is its best.

### 12.4 Where the final model still lacks (a29, ×1)

1. **Structured exact-inverse cases** vs a25_full: `chromatic_aberration` −1.55, `small_artifacts` −1.23, `salt_pepper` −1.18, `periodic_noise` −1.12. Decoder-only noise did not fix this; the gap to a19 there is 3–6 dB.
2. **Unseen families:** `unseen_v2` 30.91 (a27_full 31.42, a25_full 31.30): `line_dropout` and `film_grain`.
3. **Mild corruptions:** `resize_blur` 33.56 still below its input (34.24).
4. **Strengths:** best DISTS of every generative model in every domain; blur (`motion_blur` +0.63 over a25_full); v3 families handled (33.45 dB).

### 12.5 Conclusions

1. **a29 is the final architecture**: best at its submission setting on PSNR, SSIM and DISTS, eligible with margin (0.86), and the only one of the two whose 200k training is reproducible.
2. **Its weaknesses are not clearly hyperparameter-shaped** (structured cases, unseen families), so expect tuning to give +0.1–0.3 dB in total, not to close the 0.84 dB gap.
3. **Noise ×0.5** is the submission setting unless tuning changes the trade-off.

## 13. Round 8: tuned final runs (a29_tuned, a29_tuned_b) and the final choice

Source: `run_round8_final.log`, `results/<model>/`. Both 200k steps, data v3, a29 network; hyperparameters from Optuna trials 32 and 37 (`HYPERPARAMETER_TUNING.md` §5).

### 13.1 Candidates at their best eligible noise scale (diversity ≥ 0.6)

| Model | Scale | PSNR | SSIM | DISTS | Diversity | unseen_v2 (×1) | Rank PSNR / SSIM / DISTS |
|---|---|---|---|---|---|---|---|
| **a29_tuned** | ×0.6 | **32.817** | **0.8744** | 0.1377 | 0.640 | **32.15** | 1 / 1 / 3 |
| **a29_tuned (chosen)** | **×0.75** | 32.741 | 0.8740 | 0.1371 | **0.772** | **32.15** | 1 / 1 / 3 |
| a29_tuned_b | ×1 | 32.601 | 0.8697 | **0.1340** | 0.649 | 31.93 | 2 / 2 / 1 |
| a29 | ×0.5 | 32.269 | 0.8682 | 0.1360 | 0.86 | 30.91 | 3 / 3 / 2 |
| a19 (regression, not allowed) | – | 33.112 | 0.8743 | 0.1293 | 0 | 31.88 | – |

Average rank (PSNR, SSIM, DISTS): a29_tuned 1.67, a29_tuned_b 1.67, a29 2.67. **Tie-break (`PLAN.md` rule): better `unseen_v2` → a29_tuned** (32.15 vs 31.93). a29_tuned also wins on low-light (37.64 vs 37.43) and has the larger generative margin; a29_tuned_b wins on medical (34.83 vs 34.49) and DISTS.

### 13.2 Why noise ×0.75 and not ×0.6

×0.6 is +0.08 dB, +0.0004 SSIM but +0.0006 DISTS and only 0.04 above the diversity bar (0.640). The brief requires a generative model, and our diversity is measured on 96 of our own images, so the private set may sit lower. ×0.75 keeps 0.77 (29% margin) for −0.08 dB. It still beats a29_tuned_b at ×1 on PSNR (+0.14) and SSIM (+0.004).

### 13.3 What tuning changed (a29_tuned vs a29, same network, data and steps)

| | a29 | a29_tuned | Δ |
|---|---|---|---|
| PSNR at eligible scale | 32.27 (×0.5) | 32.74 (×0.75) | +0.47 |
| PSNR at ×1 | 31.93 | 32.61 | +0.69 |
| SSIM at eligible scale | 0.8682 | 0.8740 | +0.006 |
| DISTS at eligible scale | 0.1360 | 0.1371 | +0.001 |
| `unseen_v2` (×1) | 30.91 | 32.15 | **+1.24** |
| PSNR after epoch 40 | fell 0.15 dB | **rose 0.24 dB** | |

The higher learning rate (7.55e-4) and lower adversarial weight (0.00275) let the model keep improving fidelity while the DISTS loss acts; a29's PSNR declined in that phase.

### 13.4 Whole project: what each stage was worth (submittable setting, benchmark v1)

| Stage | Model | PSNR | SSIM | DISTS | Step |
|---|---|---|---|---|---|
| first eligible GAN (60k) | a25 | 30.93 | 0.8372 | 0.1674 | – |
| long training (200k) | a25_full | 32.02 | 0.8576 | 0.1399 | +1.09 dB |
| per-sample loss, data v3, gated decoder noise | a29 | 32.27 | 0.8682 | 0.1360 | +0.25 dB |
| Optuna tuning | **a29_tuned** | **32.74** | **0.8740** | 0.1371 | +0.47 dB |
| regression reference (not allowed) | a19 | 33.11 | 0.8743 | 0.1293 | gap −0.37 dB |

### 13.5 Conclusions

1. **Final model: a29_tuned at noise ×0.75**: 32.74 dB / 0.8740 / 0.1371, 4.98 GFLOPs, diversity 0.77.
2. It is within 0.37 dB and 0.0003 SSIM of the best non-generative model we trained, and better than it on unseen families.
3. Remaining weaknesses for the report: `resize_blur` below its input, structured artifacts 2–3 dB behind a19, DISTS 0.008 behind a19.
