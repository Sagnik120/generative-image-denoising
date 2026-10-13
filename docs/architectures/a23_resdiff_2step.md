# a23 Residual diffusion, width 16, 2 steps

Round 4: strictly generative models. Plan: `../PLAN.md`.

**How it works:** the same residual diffusion as a22 (`src/common/resdiff.py`) with a narrower network run twice: width 16, 2 sampling steps (noise levels 4 then 2). **4.61 GFLOPs in total, 2.58 M parameters.**
**Losses:** composite loss on the clean-image estimate. Because the second step works on the model's own intermediate result, half the batches build `x_t` from the network's own first estimate instead of the true clean image (self-prediction training, as in a10).
**Why it was tried:** a true multi-step sampler inside the 5 GFLOPs budget. Its second step receives a noisy intermediate image, so the noise cannot be ignored. The cost is capacity: each pass has less than half of a22's FLOPs (the size study put width 16 at -0.49 dB for regression).
**What to look for:** diversity (should be clearly above zero); gap to a22 at 1 step; whether two small steps beat one big one on DISTS.
**Data and schedule:** degradation pipeline v2 (22 families), three domains, batch 16, AdamW 4e-4 cosine, EMA 0.999, mixed precision, 60k steps, trained from scratch (no a19 weights).

## Results
Pending. After the run, fill in from `results/a23_resdiff_2step/metrics/final_evaluation_summary.json` (PSNR / SSIM / DISTS / GFLOPs, groups, domains, benchmark v2) and `generative_eval.json` (diversity at 1 and 2 steps). Reference: a15 (regression, same data and steps) 32.03 dB / 0.8594 / 0.1537; `unseen_v2` 30.99 dB.
