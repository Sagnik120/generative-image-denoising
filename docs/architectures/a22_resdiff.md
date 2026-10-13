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
Pending. After the run, fill in from `results/a22_resdiff/metrics/final_evaluation_summary.json` (PSNR / SSIM / DISTS / GFLOPs, groups, domains, benchmark v2) and `generative_eval.json` (diversity per step count). Reference: a15 (regression, same data and steps) 32.03 dB / 0.8594 / 0.1537; `unseen_v2` 30.99 dB.
