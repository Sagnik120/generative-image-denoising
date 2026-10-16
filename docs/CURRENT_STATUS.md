# Current status

Last updated 2026-10-16 (after round 6). Next steps: `PLAN.md` (round 7: final long runs). Model numbers: `COMPARISON.md` §9–11, `architectures/`.

## 1. Summary

- **We have a strong compliant submission: a25_full** (GAN + wavelet discriminator, 200k steps). At noise ×0.5: **32.02 dB / 0.8576 / 0.1399 at 4.98 GFLOPs, diversity 1.04 grey levels.**
  - 3rd of all 25 models overall.
  - Equal to the 60k regression model a15 on PSNR, better on DISTS (2nd of all models) and on unseen families (31.30 vs 30.99).
  - −1.1 dB vs a19 (regression, 200k), which is not allowed.
- **Round 6:** a27 (full per-sample loss + active diversity constraint) is +0.47 dB over a25 at 60k, the best noise scale. a28 (adversarial from 50%) ties it and is worse on low-light and unseen; dropped.
- **What still costs PSNR:** forcing the noise to matter (~0.7 dB, measured with no adversarial term in a28's first phase). It hits structured corruptions (`small_artifacts`, `periodic_noise`, `salt_pepper`, blurs) and over-processes mild ones (`resize_blur` below the input, `jpeg` ≈ input). The adversarial term itself costs nothing.
- **Next (round 7, `PLAN.md`):** two 200k runs in parallel: **a27_full** and **a29** (a27 with noise only in the decoder, gated per pixel). Then a noise-scale choice and export.
- Time: 2026-10-16. Code and runs until 10-18; 10-19 buffer for report and submission.

## 2. The brief, line by line

Every statement in `Mini Competition.pdf` that constrains us, and where we stand.

| § | Requirement | Status |
|---|---|---|
| Title, 1 | "A **small generative** model" that "denoises images and removes small artifacts" | **Not met.** a19 is regression. "Small" also signals that FLOPs matter |
| 1 | General-purpose; "not merely memorize how to reverse a small, fixed set of corruption functions"; "foundation-model problem", generalise across content and degradation type | Partly. 22 training families, random chains; generalisation to unseen families is narrow (+0.1 dB from data changes alone) |
| 2 | Input: one corrupted RGB image 256×256×3 | Met by the pipeline. For a generative model: noise, time step and all sampling steps must be created *inside* the model call; the only input is the image |
| 2 | Output 256×256×3, **uint8 in [0, 255]**; code must convert if the model output is not uint8 | Export script does this; to re-verify on the new model |
| 3.1 | Any data we have the right to use; we generate our own pairs | Met. DIV2K, Flickr2K, BSDS, COCO, LOL, NIH X-ray, IXI MRI, MedMNIST: standard research datasets. List them with licences in the report |
| 3.2 | Nine listed families + mixed; "held-out set may contain other types"; train on a broad, varied set of corruptions and severities | Met for the listed ones (all trained and benchmarked, 17 mixes). "Other types": 13 extra families, more planned |
| 3.3 | No compute limit; model size and training duration are our choice and are judged through FLOPs | Size study done (1.3 / 2.3 / 5 GFLOPs) |
| 4 | ~100 private images; natural, low-light, medical (X-ray, MRI, etc.); undisclosed degradations and severities | All three domains in training and in our benchmarks |
| 5 | FLOPs by fvcore / ptflops on a (1, 3, 256, 256) tensor, lower is better | a19: 4.98 GFLOPs (fvcore). **A multi-step sampler is counted once per step**, so the step count multiplies FLOPs. ptflops count still to check |
| 5 | PSNR and SSIM (skimage), DISTS (official DISTS-pytorch), full-reference, on uint8, averaged over the set | Our evaluation uses the same definitions. All three are full-reference: they reward closeness to the one true clean image, not realism in general |
| 6 | Four axes ranked independently; score depends on all; "try to be the best on each metric" | Drives the design: a generative model that keeps PSNR/SSIM high and FLOPs low (one step) |
| 7.1 | Weights in a standard format + inference code: 256×256×3 in, 256×256×3 out | Tooling exists (`export_inference.py`); to redo for the new model |
| 7.2 | Report, 1–2 pages: architecture, training objective, degradations | Not written. The training objective section must describe a generative objective |
| 8 | Only private-set performance counts | Our benchmark is in-distribution; treat our numbers as upper estimates |

## 3. What has been done

| Round | Models | Type | Outcome |
|---|---|---|---|
| 1 (old pipeline) | a01 NAFNet, a02 Restormer-lite | regression | superseded |
| | a03 pix2pix GAN | **generative (GAN)** | 26.54 dB, −0.8 dB vs a01, worse DISTS |
| | a04 conditional DDPM, 8 steps from pure noise | **generative (diffusion)** | collapsed (10 dB), 37 GFLOPs |
| 2 (fixed benchmark, 60k steps, ~5 GFLOPs) | a05–a09, a11: NAF, Restormer, attention, FiLM, wavelet, FFT | regression | all within 0.09 dB; plain NAF (a05) best |
| | a10 residual flow, 2 steps | flow-style, **deterministic** | 30.35 dB, −1.6 dB vs a05, worst DISTS |
| 3 (single changes on a05) | a12–a14 size, a15 degradations v2, a16 attention, a17–a18 DISTS weighting | regression | only v2 data accepted |
| Final | a19 = a15 config, 200k steps | regression | 33.11 dB / 0.8743 / 0.1293 at 4.98 GFLOPs; best of all, **not compliant** |
| 6 | a25_full (200k), a27 full per-sample loss + diversity constraint, a28 two-phase | **strictly generative, eligible** | a25_full 32.02 / 0.8576 / 0.1399 (best compliant); a27 +0.47 dB over a25 at 60k |
| 5 (60k, v2, 8 images × 2 draws) | a24 GAN + stochastic recipe, a25 + wavelet D, a26 diffusion + GAN | **strictly generative, eligible** | diversity 1.5–1.8; −0.9 to −1.0 dB vs round-4 twins; best a25 30.93 dB |
| 4 (60k, v2 data, from scratch) | a20 GAN, a21 GAN + wavelet D, a22 residual diffusion 1 step, a23 residual diffusion w16 2 steps | **strictly generative by design** | GANs −0.15/−0.23 dB vs a15 but noise ignored; a22 −0.52 dB, diversity 0.50; a23 −2.1 dB. None satisfactory yet |

29 configurations; 16 were regression models. The generative requirement was recorded as a question for the instructor in both earlier status files and never treated as a constraint. That was the error, and it is why the best result is unusable as it stands.

## 4. What the earlier generative attempts actually tell us

None of the three was a fair test of its family, so they should guide the new work without being read as "generative models lose 1.5 dB".

| Model | What it did | Why it failed | What to take from it |
|---|---|---|---|
| a03 GAN | L1 + adversarial (pix2pix, λ=100), 3-layer PatchGAN, trained from scratch, old data, no perceptual loss | Adversarial loss from step 0 on an untrained generator; weak discriminator; no SSIM/DISTS terms to anchor it | The standard recipe is the opposite order: train the generator for reconstruction first, then add a small adversarial term (SRGAN, ESRGAN, Real-ESRGAN) |
| a04 DDPM | Noise-prediction, 8 steps, starting from pure Gaussian noise, batch 6 | Starting from noise throws away the degraded image, which is already at 24.5 dB; 8 steps cannot bridge noise → image; 8× FLOPs | Start the generative process from the degraded image, not from noise |
| a10 flow | Straight path degraded → clean, clean-prediction, 2 passes at 2.5 GFLOPs each, shallow layout ([1,1,2,2]) | Each pass has half the budget and half the depth. It scores **below a13**, a single 1.3 GFLOPs pass (30.35 vs 30.87 dB). Its worst cases (`small_artifacts` −6.8, `salt_pepper` −6.2) are the capacity-limited ones from the size study. Also has no noise input, so it produces one fixed output | Splitting FLOPs over steps is very expensive. Use one full-size step. Add stochasticity to make it a real generative model |

## 5. What carries over

Findings that do not depend on the model being regression or generative:

| Asset / finding | Use in the generative phase |
|---|---|
| Data (9 datasets, 3 domains) and mixed sampler | unchanged |
| Degradation pipeline v2 (22 families, mild-skewed severity, 8% clean, shuffled chains) | unchanged; v3 extension still planned |
| Benchmarks v1 (396 images, 33 cases) and v2 (96 images, unseen families) | unchanged; all old numbers stay comparable |
| Loss set: Charbonnier + 0.05 FFT + 0.2 SSIM + 0.1 DISTS (late start) | stays as the reconstruction part of the generative objective |
| NAF U-Net backbone, width 24, [2,2,2,4]/3/[1,1,1,2] | becomes the generator / flow network |
| Architecture extras (attention, FFT, wavelet, FiLM labels) are worth ≤0.1 dB | do not revisit |
| Size: 2.3 GFLOPs costs 0.49 dB, 1.3 costs 1.05 dB; cost is per pass | one step at full size; never split the budget across steps |
| Training length: 60k → 200k steps = +1.08 dB | final generative model needs the long schedule |
| Coverage generalises, robustness does not (+3–8 dB on added families, +0.1 dB on unseen) | keep widening the degradation list |
| DISTS weight 0.1, late start; heavier weights cost PSNR/SSIM | same weights; the adversarial weight needs the same kind of sweep |
| a19 (33.11 dB) and a15 (32.03 dB) | non-generative upper references only. Not used as initialisation by default, so the submitted model is generative from its first step (optional time-saver, `PLAN.md` §8) |
| Tooling: train / evaluate / compare / export, `run_plan.py`, FLOPs counted over the whole sampler call (done for a10) | reuse; trainer already logs generator/discriminator losses |

## 6. Reference numbers

Benchmark v1 (mean input PSNR 24.49 dB) and the `unseen_v2` group of benchmark v2.

| Model | Type | PSNR | SSIM | DISTS | GFLOPs | unseen_v2 PSNR |
|---|---|---|---|---|---|---|
| a19 (200k) | regression, upper reference | 33.11 | 0.8743 | 0.1293 | 4.98 | 31.88 |
| a15 (60k) | regression, 60k reference | 32.03 | 0.8594 | 0.1537 | 4.98 | 30.99 |
| a12 (60k, v1 data) | regression, small | 31.44 | 0.8468 | 0.1682 | 2.28 | 30.39 |
| a10 (60k, v1 data) | deterministic 2-step flow | 30.35 | 0.8238 | 0.1976 | 5.04 | – |
| a20 (60k) | GAN, diversity 0.07 | 31.89 | 0.8554 | 0.1534 | 4.98 | 30.48 |
| a21 (60k) | GAN + wavelet D, diversity 0.06 | 31.81 | 0.8551 | 0.1544 | 4.98 | 30.71 |
| a22 (60k) | residual diffusion 1 step, diversity 0.50 | 31.51 | 0.8499 | 0.1654 | 5.02 | 29.72 |
| a23 (60k) | residual diffusion w16 2 steps, diversity 1.26 | 29.93 | 0.8115 | 0.2119 | 4.61 | 28.18 |
| a24 (60k) | GAN + stochastic recipe, diversity 1.69 | 30.91 | 0.8349 | 0.1666 | 4.98 | 29.68 |
| **a25 (60k)** | **GAN + wavelet D, diversity 1.54; best eligible** | **30.93** | **0.8372** | **0.1674** | 4.98 | **29.91** |
| a26 (60k) | diffusion + GAN, diversity 1.79 | 30.64 | 0.8299 | 0.1775 | 5.02 | 28.46 |
| **a25_full (200k)** | **GAN + wavelet D, diversity 1.34 (×1) / 1.04 (×0.5); best compliant** | **32.02 (×0.5)** | **0.8576** | **0.1399** | 4.98 | **31.30** |
| a27 (60k) | full per-sample loss + diversity constraint | 31.24 (×1) / 31.39 (×0.5) | 0.8489 | 0.1601 | 4.98 | 30.45 |
| a28 (60k) | a27, adversarial from 50% | 31.15 / 31.45 | 0.8511 | 0.1582 | 4.98 | 30.17 |

Target for the generative model: as close to a19 as possible on PSNR/SSIM, equal or better on DISTS, at ≤5 GFLOPs in one step.

## 7. Risks

1. **a29 is a new mechanism** (noise only in the decoder, per-pixel gate) trained straight at 200k without screening. If it fails, a27_full or a25_full is the submission; both runs share the schedule, so they are compared epoch by epoch.
2. **Diversity margin:** a27 at ×0.5 has 0.64; long training lowered a25's diversity (1.54 → 1.34). The final noise scale is chosen with diversity ≥ 0.6 as a margin.
3. **Time:** both 200k runs take ~22–23 h and must start on 10-16 afternoon to finish on 10-17 afternoon. No room for a further training round.
4. **Mild-corruption over-processing** (`resize_blur` below input) and **unseen degradations** (`glass_blur`, `film_grain`) remain.
5. **Our benchmark is in-distribution**; the private set is ~100 images, so small differences (< 0.1 dB) may reverse there.

## 8. Checklist

- [x] Data, degradations v2, benchmarks v1/v2, tooling; regression reference a19
- [x] Rounds 4–6: generative models; a25_full = first strong compliant model
- [x] Round 7 code: data v3 (4 families, 12% clean, v3 check set), a27_full, a29 (decoder-only gated noise) (CPU-tested)
- [ ] a27_full and a29 at 200k on data v3 (10-17 → 10-18)
- [ ] Optuna tuning (10-18 → 10-19) and tuned 200k run (10-19 → 10-20)
- [ ] Noise-scale choice (grid 0.5–1.0), final pick (10-18)
- [ ] Export with fresh noise per call; fvcore + ptflops; CPU test; diversity proof (10-18)
- [ ] 1–2 page report (10-19)
