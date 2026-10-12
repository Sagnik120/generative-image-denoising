# Model comparison (a05–a19)

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
