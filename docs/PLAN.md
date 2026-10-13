# Plan: strictly generative model (GAN and residual diffusion)

Written 2026-10-14. Replaces the 2026-10-11 plan. Background: `PROJECT_REVIEW_AND_OPTIONS.md` (what was done, options, sources) and `CURRENT_STATUS.md`.

**Status: proposal for approval. No code is changed until you approve.**

---

## 1. Decisions this plan follows

1. Model families: **conditional GAN** and **residual diffusion** first. **Diffusion + GAN** only if needed (§6.4).
2. Frequency: **wavelet-domain discriminator and wavelet loss for the GAN**. No network that runs entirely in the wavelet or FFT domain.
3. Losses: keep the **composite loss**; add only promising extras that cost no inference FLOPs and little code.
4. FLOPs: **about 5 GFLOPs, never above ~5.1** for a submission. Smaller is allowed later.
5. **Train from scratch.** a19's weights are not used. a19 and a15 are only reference numbers.
6. Strictly generative: noise input, generative training loss from the first step, and a check that the noise really changes the output.

---

## 2. Is ~5 GFLOPs a sensible size?

Short answer: it is small by research standards and a defensible middle position for this competition, but nobody can know its rank without seeing the other teams. Reasons:

- **Compared with published restoration networks at 256×256** (approximate, from the papers; check before quoting): DnCNN ≈ 35 GFLOPs, NAFNet-32 ≈ 16, NAFNet-64 ≈ 65, Restormer ≈ 140. Diffusion restorers are far higher because of their steps. At 5 GFLOPs we are 3–30× below the usual baselines a team would start from.
- **Our own size study** (regression, 60k steps): 5.0 → 2.3 GFLOPs cost 0.49 dB, 0.008 SSIM, 0.014 DISTS; 2.3 → 1.3 cost a further 0.57 dB. The price per GFLOP removed triples below 2.3. So 5 is on the flat part of the curve and ~2.3 is the lowest sensible point.
- **Scoring**: FLOPs is one of four ranked axes; the other three are quality and move together. Giving up 0.5 dB loses rank on three axes against every team within 0.5 dB of us, and gains FLOPs rank only against teams between 2.3 and 5 GFLOPs.
- **Generative models pay extra for small size**: a generative model already loses some PSNR against regression, so starting it at half capacity stacks two losses.

Plan: develop at ~5 GFLOPs. Once a final model exists, train its width-16 sibling (2.3 GFLOPs) as an optional second candidate (§8, step 4). The choice between them can then be made with real numbers, and with whatever is known about the other teams by then.

---

## 3. Data (kept exactly as it is)

Nothing changes in screening, so new results stay comparable with a15/a19.

- **Clean images**: the ten sources and sampling weights in `src/common/dataset.py` (natural 70%, medical 20%, low-light real + 10% synthetic darkening), 256×256 crops, flips/rotations, 4% grey.
- **Corruptions**: pipeline **v2** (22 families: 9 listed + 13 extra; 0–4 per image with probabilities 8/34/36/17/5%; severity Beta(1.0, 1.8); 50% random order; 65% of picks from the nine listed families).
- **Testing**: benchmark v1 (396 images, 33 cases) and benchmark v2 (96 images; unseen families film grain, glass blur, line dropout).
- **Later, on the final model only** (§8 step 3): widen the corruption list (v3), because coverage was worth 3–8 dB per added family.

---

## 4. Shared parts of both models

**Backbone (generator / denoiser).** The NAF U-Net already used for a05/a15/a19: width 24; encoder [2, 2, 2, 4] NAF blocks at 256/128/64/32 px; 3 blocks at 16 px; decoder [1, 1, 1, 2]; channels 24-48-96-192-384; output = input + predicted correction. 4.98 GFLOPs, 5.5 M parameters. Rounds 2–3 showed nothing beats it at this size.

**Training setup (unchanged).** Batch 16, AdamW, learning rate 4e-4 with cosine decay to 1e-6, 1k warm-up steps, EMA 0.999 of the generator, mixed precision, gradient clipping 1.0. Screening runs: 60k steps (same as a15, so directly comparable). Final run: 200k steps.

**Composite loss (unchanged, applied to the model's clean-image output).** Charbonnier × 1.0 + FFT × 0.05 + (1 − SSIM) × 0.2 + DISTS × 0.1 (DISTS on from 50% of training, on 4 images per batch). It suits both models: both output a clean-image estimate that these terms can be applied to.

**Inference wrapper.** One module whose `forward(image)` draws seeded noise, runs every sampling step, and returns the image. FLOPs are counted on this module, so nothing is hidden.

**New measurement: sample diversity.** Run each benchmark image with 8 seeds; report the mean per-pixel standard deviation (in grey levels of 255). This is the proof that the model is generative and not a relabelled regression network. Threshold used for eligibility: ≥ 0.5.

---

## 5. Model 1: conditional GAN

### 5.1 Design

```
                 noise (new each call)
                    |  added inside every block, with a learned strength
                    v
corrupted image --> NAF U-Net generator --> restored image
      |                                          |
      +------------------+-----------------------+
                         v
              discriminator (training only)
        "is this the true clean image for this input?"
```

- **Generator.** The backbone of §4 plus **noise injection**: after each NAF block, add one channel of Gaussian noise multiplied by a learned per-channel strength (starts at zero, so training begins stable and the network learns how much randomness to use). Idea from StyleGAN (Karras et al., CVPR 2019). Cost: additions only, FLOPs unchanged at ~4.98.
- **Discriminator (training only, zero inference FLOPs).** U-Net with spectral normalisation, as in Real-ESRGAN (Wang et al., ICCVW 2021): it returns a real/fake score for every pixel, which gives the generator local feedback and is stable on mixed corruptions. It is **conditional** (Pix2Pix, Isola et al., CVPR 2017): it sees the corrupted input together with the candidate, so it judges "right restoration of this input", not just "looks like a photo".
- **Why this should not repeat a03's failure.** a03 used a small 3-layer PatchGAN, L1 only, no SSIM/DISTS anchor, full adversarial weight from step 0, and old data. Each of those is changed here.

### 5.2 Losses

Included from the start:

- **Composite loss** (§4), unchanged. It anchors PSNR, SSIM and DISTS.
- **Adversarial loss, hinge form** (Miyato et al., "Spectral Normalization for GANs", ICLR 2018). On from step 0 with its weight ramped up linearly over the first 5k steps, so the generator is trained adversarially from the beginning but the first chaotic steps are damped. Weight: small. The pixel losses are around 0.02–0.05 in size while the adversarial term is around 1, so the starting weight is 0.005, to be fixed in a short smoke run so that the adversarial gradient is roughly 10–20% of the reconstruction gradient.
- **R1 gradient penalty on the discriminator**, applied every 16 steps (Mescheder et al., ICML 2018). Keeps the discriminator from overpowering the generator across our three very different domains. A few lines of code.

Frequency add-on (variant a21, §5.3):

- **Wavelet-domain discriminator.** The discriminator's input is the three high-frequency Haar bands (horizontal, vertical, diagonal detail) of the candidate and of the corrupted input, instead of RGB. Noise, blur and compression artifacts all live in those bands, so the discriminator concentrates on exactly what distinguishes a good restoration.
- **Wavelet fidelity loss.** L1 between the Haar bands of output and target, weight 0.05, added to the composite loss.
- Source for both: Korkmaz, Tekalp & Dogan, "Training Generative Image Super-Resolution Models by Wavelet-Domain Losses Enables Better Control of Artifacts", CVPR 2024. They report fewer invented textures and a better PSNR-vs-perceptual balance than RGB or Fourier losses.
- Suitability: yes. It is training-only (no FLOPs), the Haar transform is already in our code (`haar_dwt`), and it targets our main worry about GANs, losing PSNR to invented detail. One limit to know: a discriminator that sees only high-frequency bands cannot judge colour or brightness errors; those remain the job of the composite loss, which is why that loss stays.

Held back as optional (only if the GAN shows the specific problem):

- **LDL artifact loss** (Liang et al., CVPR 2022): extra penalty where the output looks like an artifact. Use if the GAN's PSNR is more than ~0.5 dB below the diffusion model or artifacts are visible.
- **Discriminator feature-matching loss** (pix2pixHD, Wang et al., CVPR 2018): use if training is unstable.
- **Focal frequency loss** in place of the plain FFT term (Jiang et al., ICCV 2021): a tuning experiment for the final model, not for screening.

Not adding several of these at once is deliberate: if five new losses go in together and the result changes, we cannot tell which one did it.

### 5.3 Two GAN runs

- **a20_gan**: generator + RGB U-Net discriminator + composite + hinge + R1.
- **a21_gan_wavelet**: same, with the wavelet-domain discriminator and the wavelet fidelity loss.

The pair measures exactly what the wavelet idea is worth.

---

## 6. Model 2: residual diffusion

### 6.1 Idea

Ordinary diffusion starts from pure noise and needs many steps; our a04 did that and collapsed. Residual diffusion (ResShift, Yue et al., NeurIPS 2023) defines the noising process so that it ends at **corrupted image + a little noise**. The reverse process then only has to remove the difference between corrupted and clean, which needs very few steps.

```
Training (forward process, T = 4 levels):
  clean x0 --------------------------------------------> corrupted y + noise
  level t:   x_t = x0 + eta_t * (y - x0) + kappa * sqrt(eta_t) * noise
             eta goes 0 -> 1, so x_t slides from clean to (corrupted + noise)

  network( x_t , y , t )  -->  estimate of clean x0      (loss on this estimate)

Sampling (reverse, K steps):
  start: x = y + kappa * noise
  repeat K times: estimate x0, move x part of the way towards it, add a little fresh noise
  output: last estimate of x0
```

### 6.2 Design

- **Network.** The backbone of §4 with two small changes: (a) input is 6 channels, the current noisy image `x_t` and the corrupted image `y` side by side (the conditioning used by SR3 and ResShift); (b) a **time embedding** tells each block which noise level it is at (the layers for this already exist in our code and were used by a10). Output = `y` + predicted correction = estimate of the clean image. FLOPs ≈ 5.0 per step.
- **Clean-image prediction**, not noise prediction. This is what ResShift does, it is stable with few steps, and it lets the composite loss act directly on the output. It is the main fix compared with a04.
- **Noise schedule.** T = 4 levels, noise scale κ = 0.1 on a 0–1 pixel scale (a starting value; one of the two things to tune later, see §6.3).
- **Stochastic sampler** from ResShift, with seeded noise, inside the model call.

### 6.3 Losses

Included from the start:

- **Composite loss on the clean-image estimate at a random level t.** This is the diffusion training objective (regress the clean image from a noised point) with our metric-matched terms in place of plain MSE. Half of each batch is at the last level (the starting point of sampling, always used at inference), half at random levels.
- **Self-prediction training** (already in a10's code): for half the batches, build `x_t` from the network's own first estimate instead of the true clean image, so that later sampling steps are trained on the kind of input they will actually receive. Matters only when more than one step is used.

Frequency add-on for diffusion:

- **Wavelet fidelity loss** (same L1 on Haar bands, weight 0.05). It has no discriminator to pair with here, so its expected effect is smaller than for the GAN. To avoid confusing the comparison it is **not** in the screening run; it is added to the final model only if a21 shows the wavelet loss helps.

Held back as optional:

- **Min-SNR loss weighting** (Hang et al., ICCV 2023): balances noise levels. With only 4 levels and clean-image prediction the benefit is small; use only if training is slow or unstable.
- **Focal frequency loss**: as for the GAN, a final-stage tuning experiment.
- **Adversarial loss on the sampled steps**: that is the diffusion + GAN model, §6.4.

### 6.4 Two diffusion runs, and the FLOPs question

One step costs ~5 GFLOPs with the full backbone. To respect the ~5 GFLOPs limit there are two ways to run it, and they answer different questions:

- **a22_resdiff**: width 24. Submittable setting: **1 step (5.0 GFLOPs)**. The same checkpoint is also evaluated at 2 and 4 steps (10 and 20 GFLOPs) for information only: it shows what more sampling would buy.
- **a23_resdiff_2step**: width 16, **2 steps (≈ 4.6 GFLOPs)**. A true multi-step sampler inside the budget.

Known risk, stated plainly: in a22 at 1 step the network sees a clean copy of `y`, so it can learn to ignore the noise. Its output would then be the same for every seed, which fails the diversity check and is not strictly generative. a23 does not have this problem (its second step works on its own noisy intermediate), but each of its steps has less than half the capacity.

**When diffusion + GAN is needed.** If a22 at 1 step fails the diversity check **and** a23 is clearly worse than the GANs, the next model is a22 with an adversarial loss on its sampled step (Denoising Diffusion GAN, Xiao et al., ICLR 2022), reusing the discriminator from §5. That forces the single step to use its noise. It is one extra run and is not started unless this condition is met.

---

## 7. How the winner is chosen

For every run: PSNR, SSIM, DISTS on benchmarks v1 and v2 (overall, by group, by domain), GFLOPs, diversity.

1. **Eligible** = diversity ≥ 0.5 grey levels and total FLOPs ≤ 5.1 GFLOPs.
2. Among eligible configurations, rank PSNR, SSIM, DISTS and FLOPs separately (as the competition does); best average rank wins. Ties go to the better unseen-family (v2) PSNR.
3. Reject a model that is more than 0.5 dB below the others on medical or low-light images (sign of adversarial artifacts).
4. Reference only: a15 (32.03 dB / 0.8594 / 0.1537 at 60k steps) shows how much the generative requirement costs.

What to expect, so results can be read sensibly (expectations, not promises): eligible models roughly 0.3–1.0 dB below a15 in PSNR, DISTS equal or better; a21 a little better than a20; a23 below a22 in PSNR.

---

## 8. Steps and time

**Step 0. Code (after approval), then 2k-step smoke runs of each model.**
- Noise injection in the generator; U-Net spectral-norm discriminator (RGB and wavelet input); hinge loss, R1, ramp.
- Residual-diffusion training loop and seeded few-step sampler (extends a10's bundle).
- Wavelet fidelity loss.
- Diversity measurement and multi-step evaluation; new entries in `run_plan.py`.
- Smoke-run checks: losses fall, sampler is reproducible with a fixed seed, FLOPs are as expected, adversarial weight calibrated.

**Step 1. Screening: a20, a21, a22, a23.** 60k steps each, from scratch, v2 data. About 5–7 h each; in parallel on separate GPUs about 7 h in total, otherwise about 24 h one after another. Then apply §7. Run the diffusion + GAN model only if the §6.4 condition is met (one more run).

**Step 2. Final model.** The winner, from scratch, 200k steps (about 17–24 h). The long schedule gave regression +1.08 dB, so this step is not optional for a good result.

**Step 3. Improvements, as time allows, in this order.**
1. Clean-input and mild-severity check (1 h, no training): a generative model must not repaint already-clean images.
2. One tuning run on the final model: the held-back loss that matches the observed weakness (LDL if PSNR is low, focal frequency if texture is weak), and test-time noise scale.
3. Corruption list v3 as a fine-tune of the final model (about 12–15 h).

**Step 4. Optional small sibling.** Same model at width 16 (2.3 GFLOPs), about 12 h, if a lower-FLOPs entry is wanted (§2).

**Step 5. Submission.** Export the sampler module, uint8 output, FLOPs with fvcore and ptflops, CPU test from a clean checkout; report (architecture, training objective, degradations) with the screening table, diversity numbers and the unseen-family results.

**If the deadline is close:** Step 0 → a21 and a22 only → Step 2 → Step 5. That is two short runs and one long run.

---

## 9. What I need from you

- Approval of this plan, or the changes you want.
- How many days remain and how many GPUs can run in parallel. This decides between the full screening (four runs) and the short path (two runs).
