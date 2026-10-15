# Architectures explored

Round 1 (a01–a04): DIV2K-only pipeline, 11–37 GFLOPs, different validation set → **not comparable** with round 2.
Round 2 (a05–a11): multi-domain data, ~5 GFLOPs each, same fixed benchmark, same losses/schedule (60k steps, bs16, AdamW lr 4e-4, EMA 0.999, AMP).

Each file: how it works, why it was tried, config, and results. Status: `../CURRENT_STATUS.md`. Model comparison: `../COMPARISON.md`. Next steps: `../PLAN.md`.

| # | File | Core idea | PSNR | SSIM | DISTS | GFLOPs | Verdict |
|---|---|---|---|---|---|---|---|
| 01 | [a01_nafnet_unet](a01_nafnet_unet.md) | NAFNet U-Net (conv) | 27.32 | 0.752 | 0.228 | 12.8 | round 1, superseded by a05 |
| 02 | [a02_restormer_lite](a02_restormer_lite.md) | Restormer-lite | 27.44 | 0.758 | 0.235 | 11.3 | round 1, superseded |
| 03 | [a03_pix2pix_gan](a03_pix2pix_gan.md) | NAF U-Net + PatchGAN | 26.54 | 0.720 | 0.246 | 12.8 | GAN hurt, dropped |
| 04 | [a04_tiny_ddpm_sr3](a04_tiny_ddpm_sr3.md) | 8-step conditional diffusion | 10.01 | 0.086 | 0.522 | 37.3 | collapsed, dropped |
| 05 | [a05_nafnet_v2](a05_nafnet_v2.md) | NAFNet at ~5 GFLOPs | 31.924 | 0.855 | 0.1542 | 4.978 | **best overall, base model** |
| 06 | [a06_restormer_v2](a06_restormer_v2.md) | Restormer everywhere | 31.682 | 0.852 | 0.1623 | 4.984 | drop |
| 07 | [a07_naf_attn_hybrid](a07_naf_attn_hybrid.md) | NAF + global attention at 16px | 31.854 | 0.8558 | 0.1563 | 4.928 | best on unseen, keep its bottleneck |
| 08 | [a08_degradation_aware](a08_degradation_aware.md) | a07 + FiLM degradation embedding | 31.835 | 0.853 | 0.1612 | 4.982 | drop |
| 09 | [a09_wavelet_naf](a09_wavelet_naf.md) | NAF U-Net in Haar domain | 31.891 | 0.8543 | 0.1545 | 5.018 | ties a05, base for low-FLOPs variant |
| 10 | [a10_resflow_fewstep](a10_resflow_fewstep.md) | 2-step residual flow | 30.349 | 0.8238 | 0.1976 | 5.039 | drop (−1.5 dB) |
| 11 | [a11_freq_spatial](a11_freq_spatial.md) | Spatial + FFT blocks | 31.838 | 0.8552 | 0.1572 | 4.884 | not carried forward |

Round-1 numbers (a01–a04) are from a different validation set; do not compare them with a05–a11.

## Round 3 (a12–a19): single-change experiments on a05, then the final run

All use one network, `src/common/plannet.py::PlanNet` (a05's NAF U-Net with an optional global-attention bottleneck and optional Haar front end). Same benchmark as round 2, so every row is comparable with a05–a11. a12–a18: 60k steps. a19: 200k steps.

| # | File | Change vs a05 | PSNR | SSIM | DISTS | GFLOPs | Verdict |
|---|---|---|---|---|---|---|---|
| 12 | [a12_naf_w16](a12_naf_w16.md) | width 16 | 31.437 | 0.8468 | 0.1682 | 2.278 | −0.49 dB for −54% FLOPs; rejected by the size rule, kept as low-FLOPs fallback |
| 13 | [a13_naf_w12](a13_naf_w12.md) | width 12 | 30.871 | 0.8364 | 0.1856 | 1.318 | −1.05 dB; too small, dropped |
| 14 | [a14_wavelet_w28](a14_wavelet_w28.md) | a09 body, width 28 | 31.359 | 0.8455 | 0.1687 | 2.522 | dominated by a12, dropped |
| 15 | [a15_naf_degv2](a15_naf_degv2.md) | degradation pipeline v2 | 32.034 | 0.8594 | 0.1537 | 4.978 | accepted; best 60k model |
| 16 | [a16_naf_attn_degv2](a16_naf_attn_degv2.md) | a15 + global attention at 16px | 31.989 | 0.8582 | 0.1550 | 4.853 | no gain, hurts `unseen_v2`; rejected |
| 17 | [a17_naf_dists0](a17_naf_dists0.md) | DISTS loss from step 0 | 31.833 | 0.8530 | 0.1517 | 4.978 | −0.0025 DISTS for −0.09 dB; rejected |
| 18 | [a18_naf_dists0_strong](a18_naf_dists0_strong.md) | DISTS 0.2, SSIM 0.3 from step 0 | 31.566 | 0.8476 | 0.1502 | 4.978 | −0.36 dB; rejected |
| 19 | [a19_final](a19_final.md) | a15 config, 200k steps | **33.112** | **0.8743** | **0.1293** | 4.978 | best; regression, not compliant; reference and initialisation for the generative model |

## Round 4 (a20–a23): strictly generative models

The brief requires a generative model, so rounds 1–3 serve as reference only. All four trained from scratch on v2 data for 60k steps with the composite loss; each has a noise input and was checked for sample diversity (`scripts/eval_generative.py`, 8 seeds, mean per-pixel std in grey levels; eligibility ≥ 0.5). Reference a15 (regression, same data and steps): 32.034 / 0.8594 / 0.1537.

| # | File | Model | Sampling | PSNR | SSIM | DISTS | GFLOPs | Diversity | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 20 | [a20_gan](a20_gan.md) | conditional GAN, RGB discriminator | 1 pass | **31.886** | **0.8554** | **0.1534** | 4.98 | 0.07 | best quality, noise ignored; not eligible |
| 21 | [a21_gan_wavelet](a21_gan_wavelet.md) | conditional GAN, wavelet discriminator + wavelet loss | 1 pass | 31.809 | 0.8551 | 0.1544 | 4.98 | 0.06 | wavelet no gain on v1 (+0.24 dB `unseen_v2`); not eligible |
| 22 | [a22_resdiff](a22_resdiff.md) | residual diffusion, width 24 | 1 step | 31.511 | 0.8499 | 0.1654 | 5.02 | 0.50 | borderline generative, −0.52 dB vs a15; 2/4 steps worse |
| 23 | [a23_resdiff_2step](a23_resdiff_2step.md) | residual diffusion, width 16 | 2 steps | 29.930 | 0.8115 | 0.2119 | 4.61 | 1.26 | generative but −2.1 dB; rejected |

Round-4 conclusion: the composite loss penalises output variance, so a GAN switches its noise off and a 1-step diffusion model removes it at a PSNR cost. Next round (a24–a27, `../PLAN.md`): loss on the mean of two samples + a diversity term + stronger adversarial signal, for both the GAN and a one-step diffusion + GAN model.

## Round 5 (a24–a26): stochastic recipe

Same backbone, data v2, 60k steps, from scratch. Batch = 8 images × 2 noise draws. Generator loss: composite on the mean of the two samples + 0.2 Charbonnier per sample + diversity hinge (2 grey levels) + hinge adversarial 0.01. Evaluated also with test-time noise ×0.5 / ×1 / ×1.5.

| # | File | Model | Frequency | PSNR | SSIM | DISTS | GFLOPs | Diversity | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 24 | [a24_gan_div](a24_gan_div.md) | a20 GAN + stochastic recipe | FFT loss, RGB discriminator | 30.907 | 0.8349 | 0.1666 | 4.98 | 1.69 | eligible; −0.98 dB vs a20, over-smoothed |
| 25 | [a25_gan_div_wave](a25_gan_div_wave.md) | a24 + wavelet discriminator + wavelet L1 | wavelet adversarial + fidelity | **30.934** | **0.8372** | 0.1674 | 4.98 | 1.54 | **best eligible**; +0.23 dB `unseen_v2` |
| 26 | [a26_resdiff_gan](a26_resdiff_gan.md) | one-step residual diffusion + GAN | FFT loss, RGB discriminator | 30.637 | 0.8299 | 0.1775 | 5.02 | 1.79 | rejected; diffusion line closed |

Round-5 conclusion: the noise now changes the output (eligible), but the recipe cost ~1 dB, of which only ~25% is the sample variance; the rest is a worse average output from the weakened per-sample reconstruction loss and the halved batch. Next (round 6, `../PLAN.md`): full per-sample composite loss + an active diversity constraint on a few extra draws (a27), a two-stage variant (a28), and a25 trained 200k steps as the compliant fallback.

## Round 6 (a25_full, a27, a28): full per-sample loss + diversity constraint, and the long run

Data v2, from scratch. a27/a28: 16 images per step + a second draw for 4 of them; composite + wavelet L1 on every output; diversity hinge 1.5 grey levels (weight 4.0); wavelet-domain discriminator, adversarial 0.005. Results pending.

| # | File | Model | Steps | Frequency | GFLOPs |
|---|---|---|---|---|---|
| 25F | [a25_full](a25_full.md) | a25 unchanged (fallback) | 200k | wavelet D + wavelet L1 + FFT | 4.98 |
| 27 | [a27_gan_wave_varctl](a27_gan_wave_varctl.md) | full per-sample loss + active diversity constraint, adversarial from step 0 | 60k | wavelet D + wavelet L1 + FFT | 4.98 |
| 28 | [a28_gan_wave_2stage](a28_gan_wave_2stage.md) | a27, adversarial from 50% (diversity from step 0) | 60k | wavelet D + wavelet L1 + FFT | 4.98 |
