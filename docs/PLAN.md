# Plan: a strictly generative model for the competition

Written 2026-10-11 (strict-generative revision). Replaces the earlier plan of the same date, which allowed a one-step model initialised from the regression network. Background and brief-by-line compliance: `CURRENT_STATUS.md`.

## 1. Overview

**Done.** Data for three domains (natural, low-light, medical), degradation pipeline v2 (22 families, random chains), two fixed benchmarks (v1: 396 images / 33 cases; v2: 96 images with unseen families), a loss set with one term per scored metric, 19 trained configurations and all tooling. Best model a19: 33.11 dB / 0.8743 SSIM / 0.1293 DISTS at 4.98 GFLOPs.

**Problem.** a19 is deterministic regression. The brief asks for a generative model. The three earlier generative attempts were weak for identifiable reasons: a03 (GAN from scratch, weak discriminator, −0.8 dB), a04 (diffusion from pure noise, collapsed), a10 (two half-size passes, no noise input, −1.6 dB).

**Decision.** Strictly generative models only (definition in §2).

**What we do now.** Screen four strictly generative candidates from scratch under the same benchmark and FLOPs budget; choose by the four ranked axes; train the winner long; reduce its sampling steps by distillation if that improves the ranks; then widen the degradations and export.

## 2. What "strictly generative" means

A model qualifies only if all four hold. Each is checked by a measurement, not by description.

| # | Criterion | Check |
|---|---|---|
| G1 | It defines a conditional distribution p(clean \| degraded) and **samples** from it: it has a random input | Noise is drawn inside the model call |
| G2 | The noise **matters**: different seeds give different outputs | Sample diversity: mean per-pixel standard deviation over 8 seeds on the benchmark must be clearly above zero (our threshold: ≥ 0.5 grey level of 255; report the number) |
| G3 | It is trained with a **generative objective** from the first step: denoising / flow matching over a noise schedule, an adversarial loss, or an ELBO | Stated in the config and the report; no reconstruction-only model is deployed |
| G4 | Inference uses the **family's sampler** (reverse diffusion steps, ODE/SDE steps, or a generator fed with noise) | The sampler module is what gets exported |

Excluded by this definition: a19 and every other regression model; a10 as built (fails G1, G2); a flow model run in one step with the degraded image supplied clean to the network and a reconstruction loss only (the optimal network ignores its noise, so it fails G2); a regression network fine-tuned briefly and relabelled.

Constraints from the brief that apply to any sampler:

- The only input is one 256×256×3 image (§2). Noise, time steps and all sampling iterations run inside `forward`.
- FLOPs are measured on that call (§5), so **K steps cost K× FLOPs**.
- The noise must be seeded; the output must be reproducible and uint8.
- PSNR, SSIM and DISTS are all full-reference (§5). No axis rewards realism that departs from the true image.
- Four axes ranked independently, "try to be the best on each" (§6).

## 3. Analysis: the price of sampling, and how to keep it small

### 3.1 Theory

1. **Regression gives the posterior mean; a sampler does not.** The output that minimises mean squared error (maximises PSNR) is E[x | y]. A model that draws exact samples x̂ ~ p(x | y) has expected squared error exactly twice the minimum, i.e. **−3.01 dB** (Blau & Michaeli, 2018). This is the worst case for a strict sampler on the PSNR axis.
2. **The loss depends on how wide the posterior is.** Where the clean image is almost determined by the input (mild noise, sparse impulses, small artifacts, chromatic shift) the posterior is narrow and mean and samples nearly coincide. Where information is destroyed (heavy blur + noise, strong JPEG, pixelation) the posterior is wide and the gap approaches 3 dB. From our benchmark: single-family cases should lose little, pair/triple cases most.
3. **There is an optimal trade-off curve between "minimum error" and "perfect realism"**, and it is reached by interpolating between the posterior mean and a posterior sample (Freirich et al., 2021). Samplers with less injected randomness sit nearer the low-error end.
4. **DISTS behaves differently from PSNR/SSIM.** DISTS compares deep-feature statistics and tolerates resampled texture (Ding et al., 2020). Posterior-mean outputs are over-smooth in wide-posterior cases; samples restore texture. So a sampler can gain on DISTS what it loses on PSNR. With three quality axes, losing two and gaining one is a net loss unless the PSNR/SSIM cost is small, so the target is: **low sampling variance, just enough to help DISTS.**

### 3.2 Consequences for design

| Design choice | Reasoning | Evidence |
|---|---|---|
| **Start the process at the degraded image plus small noise, not at pure noise** | The input is already at 24.5 dB. The generative path then only has to cover the residual, so few steps suffice and the injected variance is small | Bridge / residual-diffusion papers reach 1–15 steps where SR3 needs ~100+; our a04 (pure noise, 8 steps) collapsed |
| **Predict the clean image, not the noise** | Stable at few steps in pixel space; lets SSIM and DISTS losses act on the prediction | ResShift, InDI, Li & He (2025); a04 used noise-prediction |
| **Few steps (K = 1–4)** | FLOPs = K × per-step. Per-step capacity is expensive to give up: halving FLOPs cost 0.49 dB, quartering 1.05 dB | Our size study (a12, a13); a10 (2 × 2.5 GFLOPs) scored below a13 (1 × 1.3 GFLOPs) |
| **Small noise scale** | Sets the sampler's position on the perception–distortion curve (3.1.3) | ResShift's κ, InDI's ε play this role |
| **Adversarial term on the sampled output** | Makes few-step samples sharp and forces the network to use its noise (G2); the main tool of one- and few-step generative models | DDGAN (Xiao et al., 2022), ADD, DMD2, SinSR-style distillation |
| **Keep the image losses (Charbonnier, FFT, SSIM, DISTS)** as the form of the denoising loss on the clean-image prediction | A diffusion/flow loss is a weighted regression of the clean image from x_t; the metric is our choice. One term per scored axis worked in rounds 2–3 | a15/a19; a17/a18 for the weights |
| **Keep the NAF U-Net, width 24, v2 data, long schedule** | Backbone, data and training length were the measured levers; none depends on the objective | Rounds 2–3 |

### 3.3 What can and cannot be known in advance

Known: per-step capacity costs (size study); data and schedule effects. Unknown until measured: (a) the PSNR/SSIM cost of real sampling on our mix of narrow and wide posteriors; (b) whether sampling buys DISTS; (c) whether one stochastic step is enough for G2 or whether K ≥ 2 is needed. Step 1 is designed to measure exactly these three.

## 4. Literature review

Citations are from memory; verify venue and year before using them in the report.

### 4.1 Denoising and restoration backbones (non-generative context)

| Work | Point | Relevance |
|---|---|---|
| DnCNN (Zhang et al., TIP 2017) | Residual learning for Gaussian denoising | Residual prediction, kept |
| Restormer (Zamir et al., CVPR 2022); SwinIR (Liang et al., 2021) | Transformer restoration | Our a06: worse than NAF at equal FLOPs |
| NAFNet (Chen et al., ECCV 2022) | Plain conv block matches transformers at far lower cost | Our backbone |
| BSRGAN (Zhang et al., ICCV 2021); Real-ESRGAN (Wang et al., 2021) | Randomly shuffled degradation chains generalise to real corruption | Basis of our pipeline v2 |
| AirNet (CVPR 2022); PromptIR (NeurIPS 2023); DA-CLIP (ICLR 2024) | All-in-one restoration with degradation prompts | Small gains over a strong backbone; our a08 agrees. Not used |

### 4.2 Theory of generative restoration

| Work | Point | Relevance |
|---|---|---|
| Blau & Michaeli, CVPR 2018 | Perception–distortion trade-off; posterior sampling has 2× MMSE | Upper bound on the cost of strictness (−3 dB) |
| Freirich et al., NeurIPS 2021 | Optimal trade-off = interpolation between mean and sample | Noise scale and step count as a controlled knob |
| Ohayon et al., *Posterior-Mean Rectified Flow*, ICLR 2025 | Best error under perfect realism: predict the mean, then transport it to the image distribution | Shows hybrids are principled, but they contain a regression stage; see §5 "not chosen" |

### 4.3 Diffusion models for restoration

| Work | Formulation | Steps | Relevance |
|---|---|---|---|
| DDPM (Ho et al., NeurIPS 2020); score SDE (Song et al., ICLR 2021); EDM (Karras et al., NeurIPS 2022) | A diffusion model is a denoiser trained over noise levels | 35–1000 | Foundation: our task is literally the network's training task, generalised from Gaussian noise to arbitrary corruption |
| SR3 (Saharia et al., TPAMI 2022); Palette (SIGGRAPH 2022) | Conditional diffusion from pure noise, degraded image concatenated | 100–1000 | Too many steps; a04 was a small version and failed |
| Whang et al., *Deblurring via Stochastic Refinement*, CVPR 2022 | Regression predictor + diffusion on the residual; sample averaging trades realism for PSNR | ~10–500 | Documents the PSNR cost of sampling and how to buy it back |
| DDRM (Kawar et al., NeurIPS 2022); DPS (Chung et al., ICLR 2023); DDNM | Unconditional prior + known degradation operator | 20–1000 | Not applicable: our degradations are unknown and mixed |
| Cold Diffusion (Bansal et al., NeurIPS 2023) | Diffusion with arbitrary degradations instead of Gaussian noise | few–many | Conceptual base for "start at the degraded image" |
| InDI (Delbracio & Milanfar, TMLR 2023) | Straight path degraded → clean, small noise; one step = regression, more steps = more realism | 1–100 | Describes the step knob exactly; also shows why one deterministic step is not generative |
| I2SB (Liu et al., ICML 2023) | Schrödinger bridge between degraded and clean | ~10–1000 | Bridge formulation |
| IR-SDE (Luo et al., ICML 2023) | Mean-reverting SDE whose end state is degraded + noise | ~100 | Same idea, SDE form |
| **ResShift (Yue et al., NeurIPS 2023)** | Markov chain shifting the *residual*, end state N(degraded, κ²I), clean-image prediction | 15, later 4 | **Template for our a20**: built for few steps |
| RDDM (Liu et al., CVPR 2024) | Separate residual and noise diffusion | 5–10 | Variant of the same |
| DiffUIR (Zheng et al., CVPR 2024) | One residual-diffusion model for several degradation types | few | Closest to our all-in-one setting |
| DiffIR (Xia et al., ICCV 2023) | Diffusion on a compact prior vector guiding a restoration transformer | 4 | Efficient, but two-stage and the image path is regression; not chosen |
| StableSR, DiffBIR, SUPIR, OSEDiff (2023–2024) | Pretrained Stable Diffusion as prior | 1–50 | Roughly two orders of magnitude above our FLOPs; low PSNR; VAE harms medical detail. Rejected |

### 4.4 Flow matching and few-step / one-step generation

| Work | Point | Relevance |
|---|---|---|
| Flow Matching (Lipman et al., ICLR 2023); Rectified Flow (Liu et al., ICLR 2023) | Straight probability paths, ODE sampling, few steps | Equivalent training form for a20; deterministic sampler variant |
| DDIM (Song et al., ICLR 2021) | Fewer steps from the same trained model | K is a test-time choice: one training run gives K = 1, 2, 4 |
| Progressive Distillation (Salimans & Ho, ICLR 2022) | Halve the step count repeatedly by distillation | Step 3 |
| Consistency Models (Song et al., ICML 2023); MeanFlow (Geng et al., 2025) | One-step samplers that remain generative (noise → sample) | Step 3; a one-step model that still satisfies G1–G4 |
| DDGAN (Xiao et al., ICLR 2022) | Large denoising steps modelled by a conditional GAN: 2–4 steps | **Template for a23** |
| ADD (Sauer et al., 2023); DMD / DMD2 (Yin et al., 2024) | One-step diffusion via adversarial / distribution-matching losses | a23, step 3 |
| SinSR (Wang et al., CVPR 2024) | Distils ResShift to one step | Direct precedent for step 3 on our a20 |

### 4.5 GANs

| Work | Point | Relevance |
|---|---|---|
| pix2pix (Isola et al., CVPR 2017) | Conditional GAN, PatchGAN discriminator; noted that the generator tends to ignore input noise | a03 copied it; the noise-ignoring problem is why G2 is measured |
| SRGAN (Ledig et al., CVPR 2017); ESRGAN (Wang et al., 2018) | Adversarial + perceptual loss; lower PSNR, sharper output | Quantifies the trade; shows a short reconstruction warm-up is standard |
| Real-ESRGAN (Wang et al., 2021) | U-Net discriminator with spectral norm, stable on mixed degradations | Discriminator for a22/a23 |
| StyleGAN (Karras et al., CVPR 2019) | Per-layer noise injection drives stochastic detail | Noise injection for a22 |
| DeblurGAN-v2 (Kupyn et al., ICCV 2019) | GAN deblurring, efficient backbones | Precedent for blur |

### 4.6 Other generative families

| Family | Works | Verdict |
|---|---|---|
| VAE | DivNoising (Prakash et al., ICLR 2021), HDN | Samples are blurry, weak on all three quality metrics; rejected |
| Normalising flows | SRFlow (Lugmayr et al., ECCV 2020), Noise Flow | Invertibility forces large networks at full resolution; rejected on FLOPs |
| Autoregressive | – | One pass per pixel/token; rejected on FLOPs |

### 4.7 What the literature says, in short

- For restoration, the efficient generative models are residual/bridge diffusions and GANs; both work in 1–4 network evaluations.
- Every strict sampler pays PSNR against regression; the papers that report both (Whang et al., ResShift, SinSR, ESRGAN) show it. Few steps, small noise and sample-consistent training reduce it.
- One-step models stay generative when obtained by consistency or adversarial training from a diffusion, not by dropping the noise.

## 5. Candidates

All share: NAF U-Net (a05/a19 layout) with a time embedding and a noise path; degradation pipeline v2; batch 16, AdamW 4e-4 cosine, EMA; **trained from scratch** with the generative objective throughout; whole sampler inside `forward`.

**Diffusion formulation (a20, a21, a23)**, after ResShift. With clean x₀, degraded y, schedule 0 = η₀ < … < η_T = 1 and noise scale κ:

- forward: x_t = x₀ + η_t·(y − x₀) + κ·√η_t·ε, so x_T ~ N(y, κ²I);
- network f(x_t, y, t) predicts x₀; loss = Charbonnier + 0.05 FFT + 0.2 SSIM + 0.1 DISTS on the prediction, t sampled over the schedule;
- reverse (stochastic): x_{t−1} = (η_{t−1}/η_t)·x_t + (1 − η_{t−1}/η_t)·x̂₀ + κ·√(η_{t−1}(η_t − η_{t−1})/η_t)·ξ;
- K-step sampler: draw x_T = y + κε, apply K reverse steps on a sub-schedule. κ starts at 0.1 (on a 0–1 scale) and is one of the things tuned.

| ID | Model | Sampler | Total GFLOPs | Purpose |
|---|---|---|---|---|
| a20 | Residual diffusion, width 24 | K = 1, 2, 4 from one checkpoint | 5 / 10 / 20 | The perception–distortion–FLOPs curve of real diffusion sampling at full per-step capacity |
| a21 | Residual diffusion, width 16 | K = 2 | ~4.6 | Multi-step at **equal FLOPs** to a19; a correctly built version of a10 (stochastic, proper depth) |
| a22 | Noise-conditioned GAN, width 24 | 1 generator pass | ~5 | Strict GAN: per-layer noise injection, conditional U-Net discriminator (spectral norm), hinge loss + image losses; adversarial weight ramped in over the first 10% of training |
| a23 | Diffusion-GAN, width 24 | K = 1 and 2 | 5 / 10 | a20 plus an adversarial loss on the *sampled* x_{t−1} (DDGAN): the discriminator sees (x_{t−1}, x_t, y, t). Aims at few-step sharpness and at making the noise matter in one step |

Why these four: they cover the two efficient strict families (diffusion, GAN), their combination, and the one design question the size study cannot answer (one big step vs two smaller ones under sampling). Each differs from the next by one thing.

**Not chosen**, with reasons:

| Option | Reason |
|---|---|
| Diffusion from pure noise (SR3-style) | Many steps; a04; every bridge paper improves on it. One optional run (a24, 4 steps, width 12) only if a baseline is wanted for the report |
| Deterministic one-step flow (InDI K=1, rectified flow from y) | Equals regression; fails G1/G2 |
| Predict-then-refine (regression + small diffusion; Whang et al., PMRF) | Half the model is a regression network; conflicts with the strict decision and pays FLOPs for two networks. Fallback only if a20–a23 all lose > 1.5 dB |
| Pretrained latent diffusion | FLOPs, PSNR, medical detail (§4.3) |
| VAE, normalising flow, autoregressive | §4.6 |
| Zero-shot priors (DDRM, DPS) | Need the degradation operator |

## 6. Steps

### Step 0. Preparation (no GPU)

Code to be written and reviewed before any run (nothing is changed yet):

1. Time embedding and noise path in the shared network (a10's `ResFlowNet` already has the time embedding and the 6-channel input).
2. Residual-diffusion bundle: forward process, loss, stochastic K-step sampler module with seeded noise inside `forward`.
3. Conditional U-Net discriminator, hinge loss, ramp schedule; DDGAN-style step discriminator for a23 (the trainer already logs generator/discriminator losses from a03).
4. Evaluation: metrics at each K; **diversity** (G2); metrics of the 8-sample average (shows the mean–sample gap for the report); clean-input test.
5. FLOPs counted on the sampler module with fvcore and ptflops.
6. `run_plan.py` entries and the rules below.

Sanity checks before the long runs: a 2k-step smoke run per candidate; sampler reproducible with a fixed seed; FLOPs scale as K×.

### Step 1. Screening, round 4 (4 runs, 60k steps each, ~5–6 h each, parallel across GPUs)

a20–a23, from scratch, v2 data. 60k steps makes them directly comparable with a15 (regression, 32.03 / 0.8594 / 0.1537, `unseen_v2` 30.99), which is the yardstick for "price of being generative".

Measured for every candidate and every K: PSNR, SSIM, DISTS, GFLOPs, diversity, per group and domain, `unseen_v2`.

Rules, fixed now:

| # | Rule |
|---|---|
| R1 | **Eligibility:** satisfies G1–G4 with diversity ≥ 0.5/255. A configuration that fails G2 is not a candidate at that K, whatever its PSNR |
| R2 | **FLOPs cap:** total ≤ 5.1 GFLOPs for the default entry. Configurations above it (a20 at K=2, 4) are measured for information and as distillation teachers |
| R3 | **Selection:** among eligible configurations under the cap, rank each of PSNR, SSIM, DISTS, FLOPs separately (as the competition does) and take the best average rank; ties go to higher `unseen_v2` PSNR |
| R4 | **Domain safety:** reject a configuration that is > 0.5 dB below the candidate average in medical or low-light (adversarial artefacts) |
| R5 | **If the best eligible model is > 1.0 dB below a15:** before the long run, tune κ (0.05, 0.2) and the adversarial weight on the winner in two extra 60k runs |
| R6 | **If a20 at K=2 or 4 beats every ≤5 GFLOPs configuration by ≥ 0.5 dB or ≥ 0.01 DISTS:** it becomes the teacher for step 3 |

What each comparison answers:

| Comparison | Question |
|---|---|
| a20 K=1 vs K=2 vs K=4 | Does more sampling help DISTS enough to justify the PSNR and FLOPs? (expected: PSNR falls, DISTS improves, with K) |
| a21 (2 × 2.3) vs a20 K=1 (1 × 5) | One big step or two smaller ones at equal FLOPs |
| a22 vs a20 K=1 | GAN vs diffusion as the one-pass strict model |
| a23 vs a20 | What the adversarial term adds to diffusion at 1–2 steps, and whether it produces diversity |
| all vs a15 | The measured price of being generative on our benchmark |

Expectations (to test, not promises): eligible one-pass models 0.3–1.0 dB below a15 with DISTS equal or better; a21 below a20 K=1 by ~0.5 dB (per-step capacity); pure diffusion at K=1 may fail G2 because the network can read the clean condition and ignore the noise, in which case a23 or K=2 carries the diffusion line.

### Step 2. Long run of the winner (1 run, 200k steps, ~17–22 h)

Same configuration as the screening winner, from scratch, 200k steps (the schedule that gave regression +1.08 dB). Adversarial terms, if any, keep their screening weights. Output: a25, the first submission-grade generative model. Re-check R1 and R4.

### Step 3. Fewer steps by distillation (only if rule R6 fired or the winner uses K ≥ 2; ~8–10 h)

Distil the K-step teacher into a student of the same network run at K/2 (then K/4) steps: consistency or SinSR-style, the student still receives noise and is trained to match the teacher's sample for the same noise, with the adversarial term kept to preserve sharpness and diversity. The student remains a generative sampler (G1–G4) at a fraction of the FLOPs.

- **Rule:** accept the student if PSNR/SSIM/DISTS are within 0.1 dB / 0.002 / 0.003 of the teacher and diversity stays above the R1 threshold.

### Step 4. Improvements on the chosen model

1. **Diagnostics (1 h):** per-severity and clean-input test (a generative model must not repaint clean images: output on clean inputs > 45 dB); implementation-shift test (`skimage.util.random_noise`, OpenCV vs PIL JPEG, chroma subsampling variants).
2. **Degradation pipeline v3 (one long fine-tune, ~15 h):** more families (ImageNet-C style glass/zoom blur and spatter, film grain, line dropout, grey vs per-channel and spatially varying noise, JPEG variants, MRI/ultrasound/X-ray artifacts), more 3–4-step chains, more JPEG; new held-out families for a benchmark v3. Coverage was worth 3–8 dB per added family in round 3 and is the main lever for the private set.
3. **Noise scale and sampler tuning at test time (no training):** κ and the amount of reverse-step noise can be lowered at inference within G2; pick the point with the best average rank.
4. **Small sibling (~12 h):** the same generative model at width 16 (2.3 GFLOPs), distilled from the final model, as a second entry if allowed.

### Step 5. Submission and report

- Export the sampler module (seeded noise inside), uint8 output, standard weight format; FLOPs by both counters; CPU test from a clean checkout on a colour, a grey-replicated and a clean image.
- Report, 1–2 pages, in the brief's order: **architecture** (NAF U-Net denoiser, time and noise inputs, sampler, FLOPs); **training objective** (residual diffusion loss, adversarial loss, schedule); **degradations** (families, chains, severities). Evidence: round-4 table, curve over K, diversity, mean-of-samples vs single sample, unseen-family results, a15/a19 as non-generative reference.

## 7. Budget and order

| Order | Work | GPU time |
|---|---|---|
| 1 | Step 0 code + smoke runs | < 1 h |
| 2 | Step 1: a20–a23 in parallel | ~22 GPU-h total, ~6 h wall on 4 GPUs |
| 3 | (R5 only) two tuning runs | ~11 h |
| 4 | Step 2: long run | ~17–22 h |
| 5 | Step 3: distillation, if triggered | ~8–10 h |
| 6 | Step 4: diagnostics, v3, sibling | ~1 + 15 + 12 h |
| 7 | Step 5 | ~1 h |

Short version if time is tight: a20 and a23 only → long run of the better → step 5.

## 8. Options that need your decision

- **Initialising the long run from a19's weights** would save most of the 200k steps and likely end closer to a19's PSNR. It is common practice (SRGAN, Real-ESRGAN, SinSR start from pretrained networks) and the result would still satisfy G1–G4, but the model would then begin as a regression network. Default in this plan: **no**, train from scratch. Say if you want it.
- **Asking the instructor** whether FLOPs are counted over all sampling steps and whether a one-step sampler is acceptable. Not required to proceed: the plan assumes the strict answer to both.

## 9. Not doing

Regression models or regression-only fine-tunes; new backbones, attention, FFT, wavelet, degradation-label conditioning (≤ 0.1 dB in rounds 2–3); diffusion from pure noise as a main line; pretrained latent diffusion; models above 5 GFLOPs as the default entry or below ~2 GFLOPs.
