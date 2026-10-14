# a23 Residual diffusion, width 16, 2 steps

Round 4: strictly generative models. Plan: `../PLAN.md`.

**How it works:** the same residual diffusion as a22 (`src/common/resdiff.py`) with a narrower network run twice: width 16, 2 sampling steps (noise levels 4 then 2). **4.61 GFLOPs in total, 2.58 M parameters.**
**Losses:** composite loss on the clean-image estimate. Because the second step works on the model's own intermediate result, half the batches build `x_t` from the network's own first estimate instead of the true clean image (self-prediction training, as in a10).
**Why it was tried:** a true multi-step sampler inside the 5 GFLOPs budget. Its second step receives a noisy intermediate image, so the noise cannot be ignored. The cost is capacity: each pass has less than half of a22's FLOPs (the size study put width 16 at -0.49 dB for regression).
**What to look for:** diversity (should be clearly above zero); gap to a22 at 1 step; whether two small steps beat one big one on DISTS.
**Data and schedule:** degradation pipeline v2 (22 families), three domains, batch 16, AdamW 4e-4 cosine, EMA 0.999, mixed precision, 60k steps, trained from scratch (no a19 weights).

## Results
Benchmark v1, 60k steps, single seed (`generative_eval.json`, 8 seeds).

| Steps | PSNR | SSIM | DISTS | GFLOPs | Diversity |
|---|---|---|---|---|---|
| 1 | 30.104 | 0.8161 | 0.2093 | 2.31 | 0.73 |
| **2 (configured)** | 29.930 | 0.8115 | 0.2119 | 4.61 | **1.26** |
| 4 | 29.620 | 0.8027 | 0.2147 | 9.23 | 1.71 |
| a22 (width 24, 1 step) | 31.511 | 0.8499 | 0.1654 | 5.02 | 0.50 |
| a12 (width 16 regression, v1 data) | 31.437 | 0.8468 | 0.1682 | 2.28 | 0 |

| Group (2 steps) | single | extra | pair | triple | unseen | natural | lowlight | medical | unseen_v2 | plan (v2) |
|---|---|---|---|---|---|---|---|---|---|---|
| PSNR | 32.13 | 32.06 | 28.80 | 28.10 | 29.06 | 26.66 | 35.04 | 31.36 | 28.18 | 33.77 |

## Analysis
- **Clearly generative (diversity 1.26) but clearly the worst model of round 4:** −2.10 dB vs a15, −1.58 dB vs a22, DISTS 0.212 (worst of all 19 benchmarked models). Last in the round-4 ranking.
- **The second step hurts:** 1 step 30.10 dB → 2 steps 29.93. Self-prediction training was on, so this is not only a train/test mismatch; with a reconstruction-only loss the second step cannot add anything but its own noise.
- **Even its 1-step output (2.3 GFLOPs) is 1.33 dB below a12**, a width-16 regression at the same FLOPs trained on older data. Small capacity and a noisy input together cost far more than either alone. Worst cases: `chromatic_aberration` −6.3 dB vs a15, `small_artifacts` −5.7, `salt_pepper` −5.4, `periodic_noise` −5.4: the capacity-limited cases of the size study, as with a10.
- **Confirms round 2 (a10):** splitting the 5 GFLOPs budget across steps is the most expensive choice available. Multi-step sampling at this budget is closed.

**Verdict: rejected. No more multi-step models at ≤ 5 GFLOPs.**
