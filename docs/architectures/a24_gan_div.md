# a24 Conditional GAN with the stochastic recipe

Round 5: make the generative model really sample. Plan: `../PLAN.md` §4–5. Round-4 analysis: `../COMPARISON.md` §9.

**How it works:** a20's conditional GAN, trained so that its noise cannot be switched off.
- **Generator** (`src/common/gennet.py::GenNet`): the a05/a15 NAF U-Net (width 24, enc [2,2,2,4], mid 3, dec [1,1,1,2], residual output) with a `NoiseInjection` after every NAF block. Noise strength per channel is learned and **starts at 0.01** (a20: 0). In eval mode the noise comes from seeded buffers, so one seed gives one reproducible sample; `reseed_noise` draws another. **4.98 GFLOPs, 5.52 M parameters, one pass.**
- **Discriminator** (`src/common/gan.py::UNetDiscriminatorSN`, training only): U-Net with spectral normalisation, 32 base channels, per-pixel real/fake score; conditional, input = candidate + corrupted image (RGB, 6 channels).
- **Batch:** 8 images × 2 noise draws = 16 generator outputs per step (same compute as a20's 16).

**Losses (generator):**
| Term | Applied to | Weight | Why |
|---|---|---|---|
| Composite (Charbonnier 1.0 + FFT 0.05 + SSIM 0.2 + DISTS 0.1 from 50%) | **mean of the two samples** | 1 | keeps the average output, which PSNR/SSIM reward, on the truth; halves the variance penalty that killed a20's noise |
| Charbonnier | each sample | 0.2 | keeps every single sample anchored |
| Diversity hinge `max(0, 2/255 − mean|x1 − x2|)` | sample pair | 1.0 | pushes samples ~2 grey levels apart, then stops (DSGAN, Yang et al. ICLR 2019; Mode-Seeking GAN, Mao et al. CVPR 2019; target added here) |
| Hinge adversarial | each sample | 0.01, ramped over 5k steps | 2× a20; turns the variance into realistic detail |

Discriminator: hinge + R1 (γ = 1, every 16 steps). **Frequency:** FFT magnitude term in the composite loss; discriminator in RGB.

**Why it was tried:** a20 was stable and only −0.15 dB vs regression, but ignored its noise (diversity 0.07). a24 changes only what made it ignore the noise. a24 − a20 = the price of real sampling.
**What to look for:** diversity 1–2 grey levels (eligibility ≥ 0.5); `train_spread255` in `training_log.csv` staying near 2 (falls toward 0 = collapse); PSNR within ~0.3 dB of a20; DISTS ≤ a20; medical / low-light not worse than the overall loss; noise-scale sweep (×0.5, ×1, ×1.5).
**Data and schedule:** pipeline v2, three domains, AdamW 4e-4 cosine, EMA 0.999, AMP, 60k steps, from scratch.

## Results
Pending. Fill in from `results/a24_gan_div/metrics/final_evaluation_summary.json` and `generative_eval.json`. References: a20 31.886 / 0.8554 / 0.1534, diversity 0.07; a15 (regression) 32.034 / 0.8594 / 0.1537.
