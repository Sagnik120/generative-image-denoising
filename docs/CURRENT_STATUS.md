# Current status

Last updated 2026-10-14 (after round 4). Next steps: `PLAN.md` (round 5). Model numbers: `COMPARISON.md` §9, `architectures/`.

## 1. Summary

- **Still no compliant submission.** a19 (33.11 dB) is regression. Round 4 trained four strictly generative candidates (a20–a23); none is yet a good generative submission:
  - **a20 / a21 (GANs)**: best quality of round 4 (31.89 / 31.81 dB, only −0.15 / −0.23 dB vs the like-for-like regression a15), but they **switched their noise off** (diversity 0.07 / 0.06 grey levels, threshold 0.5). In practice deterministic.
  - **a22 (residual diffusion, 1 step)**: diversity 0.50, exactly at the threshold; −0.52 dB and worse DISTS than a15. Sampling with 2 or 4 steps is worse still.
  - **a23 (diffusion, width 16, 2 steps)**: clearly generative (1.26) but −2.1 dB; rejected.
- **Root cause found:** every term of the composite loss compares one output with one target, so output variance is pure penalty. The GAN removes its noise; the diffusion model erases its start noise and pays PSNR for it.
- **Kept:** the GAN machinery is stable and cheap in quality (−0.15 dB, no medical/low-light damage); one full-size pass is the only viable budget (a10, a22 sweep, a23 all agree).
- **Next (round 5, `PLAN.md`):** a "stochastic recipe" (reconstruction loss on the mean of two samples, diversity term with a target, stronger adversarial weight) in a GAN (a24), a wavelet-discriminator GAN (a25) and a one-step diffusion + GAN (a26); a22 (60k) kept as the safety net; then the winner trained 200k steps. 5 days of experiments, day 6 for the report.

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
| 4 (60k, v2 data, from scratch) | a20 GAN, a21 GAN + wavelet D, a22 residual diffusion 1 step, a23 residual diffusion w16 2 steps | **strictly generative by design** | GANs −0.15/−0.23 dB vs a15 but noise ignored; a22 −0.52 dB, diversity 0.50; a23 −2.1 dB. None satisfactory yet |

23 configurations; 16 were regression models. The generative requirement was recorded as a question for the instructor in both earlier status files and never treated as a constraint. That was the error, and it is why the best result is unusable as it stands.

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

Target for the generative model: as close to a19 as possible on PSNR/SSIM, equal or better on DISTS, at ≤5 GFLOPs in one step.

## 7. Risks

1. **The variance fix may not be enough** (a24/a26 still collapse). Remedies in order: raise the diversity target or adversarial weight; test-time noise scale; VAE latent noise (Option G). Safety net: the existing a22 checkpoint (60k) or the best eligible round-5 checkpoint.
2. **Quality cost of real sampling.** Variance arithmetic: ~1–2 grey levels of sample spread should cost ~0.1–0.3 dB on top of a20's −0.15 dB. To be measured in round 5.
3. **Time.** Screening (Day 1–2) plus a 200k run (~29 h for a GAN) leaves about one day of margin before day 6. No room for a second full round of screening.
4. **Unseen degradations**: round-4 generative models are 0.3–1.3 dB below a15 on `unseen_v2`; `glass_blur`, `film_grain` remain weak.
5. **Our benchmark is in-distribution** (unchanged). Model selection used benchmark v1.

## 8. Checklist

- [x] Data, degradations v2, benchmarks v1/v2, tooling
- [x] Backbone, size, loss and training-length studies (rounds 2–3)
- [x] Regression reference a19
- [x] Round 4 screening a20–a23 (GAN, wavelet GAN, residual diffusion 1 and 2 steps) and diversity measurement
- [x] Round 5 code: stochastic recipe, a24–a26, noise-scale sweep (CPU smoke-tested)
- [ ] Round 5 screening a24, a25, a26 on two GPUs (`PLAN.md` Day 1–2)
- [ ] Final long run of the winner (Day 2–4); optional width-16 sibling and v3 fine-tune
- [ ] Export with sampler inside the model call; fvcore and ptflops; CPU test; diversity proof
- [ ] 1–2 page report (Day 6)
