# a22 Residual diffusion, 1 step

Round 4: strictly generative models. Plan: `../PLAN.md`.

**How it works:** diffusion whose forward process ends at the corrupted image plus a little noise, not at pure noise (ResShift, Yue et al., NeurIPS 2023). Code: `src/common/resdiff.py`.
- **Forward process**, T = 4 levels: `x_t = x0 + eta_t (y - x0) + kappa sqrt(eta_t) eps`, with `eta = (t/4)^2` (0.06, 0.25, 0.56, 1.0) and `kappa = 0.1`. At the last level `x_T = y + 0.1 * noise`.
- **Network** (`ResDiffNet`): the NAF U-Net, width 24, with a 6-channel input (corrupted image and `x_t`) and a time embedding injected by FiLM after every stage. Output = corrupted image + correction = estimate of the clean image.
- **Sampler** (`ResDiffSampler`): start at `y + kappa * noise`, take K reverse steps with fresh seeded noise, all inside one forward call. Configured K = 1: **5.02 GFLOPs, 5.68 M parameters.**

**Losses:** the composite loss on the clean-image estimate at a random level (half the batch at the last level, half spread over all four).
**Why it was tried:** the textbook generative family for denoising, built so that few steps suffice. Fixes a04's two faults (start from pure noise, noise prediction). The same checkpoint is scored at 1, 2 and 4 steps (5, 10, 20 GFLOPs) to show what more sampling buys; only the 1-step setting is within the FLOPs limit.
**Known risk:** at 1 step the network sees a clean copy of the corrupted image, so it can learn to ignore the noise. Diversity would then be near zero and the 1-step model would not count as generative. If so, a23 or a diffusion + GAN model carries the diffusion line (`../PLAN.md` section 6.4).
**What to look for:** diversity at 1 step; how PSNR, SSIM and DISTS move from 1 to 2 to 4 steps.
**Data and schedule:** degradation pipeline v2 (22 families), three domains, batch 16, AdamW 4e-4 cosine, EMA 0.999, mixed precision, 60k steps, trained from scratch (no a19 weights).

## Results
Benchmark v1, 60k steps, single seed. The same checkpoint sampled with 1, 2 and 4 steps (`generative_eval.json`, 8 seeds).

| Steps | PSNR | SSIM | DISTS | GFLOPs | Diversity |
|---|---|---|---|---|---|
| **1 (submittable)** | 31.511 | 0.8499 | 0.1654 | 5.02 | **0.498** |
| 2 | 31.216 | 0.8423 | 0.1692 | 10.04 | 1.61 |
| 4 | 30.678 | 0.8277 | 0.1709 | 20.08 | 2.40 |
| a15 (regression ref.) | 32.034 | 0.8594 | 0.1537 | 4.98 | 0 |

| Group (1 step) | single | extra | pair | triple | unseen | natural | lowlight | medical | unseen_v2 | plan (v2) |
|---|---|---|---|---|---|---|---|---|---|---|
| PSNR | 34.21 | 33.60 | 30.10 | 29.10 | 31.11 | 28.08 | 36.52 | 33.36 | 29.72 | 36.39 |
| vs a15 | −0.68 | −0.92 | −0.43 | −0.26 | −0.50 | −0.49 | −0.42 | −0.69 | −1.27 | +0.19 |

## Analysis
- **Borderline generative at 1 step:** diversity 0.498 grey levels, just at the 0.5 threshold. The sampler starts from `y + 0.1·noise` (noise of ~25 grey levels) and removes almost all of it; what is left is a faint residue of the starting noise, not a meaningful choice between plausible images.
- **Quality cost is real: −0.52 dB, −0.010 SSIM, +0.012 DISTS vs a15**, and −1.27 dB on `unseen_v2`. Four times the PSNR cost of a20.
- **Where it loses:** the structured cases that need exact detail: `small_artifacts` −2.33, `salt_pepper` −1.30, `periodic_noise` −1.25, `chromatic_aberration` −1.09, `motion_blur` −1.04. On random-noise cases it is close to a15 (−0.2 to −0.3). Two reasons: (1) the network must also remove its own sampling noise from `x_T`, which hides fine structure it needs to see; (2) half of training is spent on levels 1–3, which a 1-step sampler never uses, so the 1-step path received about half the effective training of a15.
- **More steps make it worse, not better** (−0.30 dB at 2 steps, −0.83 at 4, DISTS also worse). In theory more steps trade PSNR for realism, but here DISTS does not improve either. The model was trained with `steps = 1`, so self-prediction training was off: later steps see intermediate images unlike anything in training. And with a pure reconstruction loss each step returns an average estimate, so re-noising and re-estimating only adds error. Nothing in the objective rewards sharper or more realistic samples.
- **What it teaches:** residual diffusion gives diversity by construction, but with reconstruction loss only it pays PSNR for noise it then throws away. To be worth its cost it needs a term that rewards realistic samples (adversarial: diffusion + GAN, a26) and training focused on the level actually sampled.

**Verdict: only marginally generative, −0.52 dB vs regression, multi-step useless as trained. Not the winner as is; base for the diffusion + GAN model a26, and the safety-net candidate because its training objective is a genuine diffusion objective.**
