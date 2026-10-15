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
Benchmark v1 (396 images, 33 cases), 60k steps, single seed. Diversity = mean per-pixel std over 8 seeds, grey levels of 255. Run time 6.8 h (incl. evaluation) on one RTX A5000.

| Noise scale | PSNR | SSIM | DISTS | GFLOPs | Diversity |
|---|---|---|---|---|---|
| ×0.5 | 30.937 | 0.8367 | 0.1647 | 4.98 | 1.20 |
| **×1 (as trained)** | **30.907** | **0.8349** | **0.1666** | 4.98 | **1.69** |
| ×1.5 | 30.814 | 0.8315 | 0.1747 | 4.98 | 1.96 |
| a20 (ref., same GAN, noise ignored) | 31.886 | 0.8554 | 0.1534 | 4.98 | 0.07 |

| Group | single | extra | pair | triple | unseen | natural | lowlight | medical | unseen_v2 | plan (v2) |
|---|---|---|---|---|---|---|---|---|---|---|
| PSNR | 33.48 | 33.05 | 29.53 | 28.61 | 30.52 | 27.46 | 35.80 | 32.90 | 29.68 | 34.58 |
| vs a20 | −1.12 | −1.66 | −0.85 | −0.60 | −1.01 | −0.99 | −0.95 | −0.98 | −0.80 | −1.40 |

Mean learned noise strength 0.0196 (a20: 0.0017). `train_spread255` (mean |x1 − x2| of the two training samples) 2.2 at epoch 1, 2.6 from epoch 12 to the end: always above the 2.0 target, so the diversity hinge was inactive for most of training. Validation PSNR gap to a20: −0.87 dB at epoch 4, −1.05 at epoch 12, −0.98 at the end.

## Analysis
- **The recipe made the GAN generative:** diversity 1.69 (a20: 0.07), noise strength 12× a20. Eligible.
- **But it cost −0.98 dB, −0.021 SSIM and +0.013 DISTS vs a20**, about 4× the 0.1–0.3 dB predicted. DISTS got worse, not better.
- **The sample spread itself explains only ~25% of the loss.** a20's error is 6.47 grey levels RMS (MSE 41.8); a24's is 7.29 (MSE 53.2). The sample variance adds 1.69² ≈ 2.9 of the extra 11.4. The other ~75% is a worse *average* output. The noise-scale sweep confirms it: halving the noise at test time removes half the variance but gains only +0.03 dB.
- **What the loss looks like in the images** (`epoch_0024_comparison.png`): no visible grain. Outputs are **over-smoothed / painterly**: fine texture (fish scales, coral, water ripples, grass) is flattened and small text ("South Tyne") is blurred. The model is less sharp than a20, not noisier.
- **Where it loses** (vs a20): `small_artifacts` −3.90, `periodic_noise` −1.65, `salt_pepper` −1.57, `chromatic_aberration` −1.36, `gaussian_blur` −1.24, `motion_blur` −1.19, `defocus_blur` −1.18, `uniform_noise` −1.08, `gaussian_noise` −0.99; least on `jpeg` −0.43 and `resize_blur` −0.16. The exact-inverse cases (structured artifacts and deblurring) suffer most, i.e. the ones where a19 gained most from longer training: they need the most precise reconstruction signal. All three domains lose about the same (−0.95 to −0.99): no adversarial artifacts on medical or low-light.
- **Easy groups lose more dB than hard ones** (single −1.12, triple −0.60): consistent with an added error floor of ~2.6–3.4 grey levels RMS on top of a20's error, whatever the corruption.
- **Why the average output got worse.** a26 lost about the same (−0.87 vs a22) without any feature-noise injection, so the injected noise in the features is not the main cause. What a24 and a26 share: (1) **8 unique images per step instead of 16**, i.e. half the images seen in 60k steps; (2) **the reconstruction gradient each sample receives is halved** (loss on the mean of two: ∂L/∂x1 = ½ L′; plus 0.2 per sample = 0.7 of a20's); (3) the adversarial weight doubled, so relative to reconstruction it is ~3× a20's. The discriminator loss stays at ~1.7 (2.0 = chance), so its gradient is weak and noisy: a stronger weight on a weak signal adds noise to the generator's updates rather than texture. Evidence for (1) alone not being enough: a20 had seen the same number of images at epoch 12 (31.65 dB, learning rate still high), already 0.74 dB above a24's final.
- **What it teaches:** variance itself is cheap (~0.25 dB at 1.7 grey levels). The expensive part was weakening the per-sample reconstruction loss and the batch. Keep the per-sample composite loss at full weight and buy the variance with a constraint (diversity hinge that is actually active), on a few extra draws instead of halving the batch. See `../PLAN.md` round 6 (a27).

**Verdict: first eligible GAN (diversity 1.69), but −0.98 dB vs a20; superseded by a25 (same quality, better `unseen_v2`) and by the round-6 recipe.**
