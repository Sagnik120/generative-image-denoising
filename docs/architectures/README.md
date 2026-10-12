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
