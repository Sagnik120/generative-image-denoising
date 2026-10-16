# a29 Decoder-only, per-pixel gated noise (Option L)

Round 7. Plan: `../PLAN.md` §4. Why: `../COMPARISON.md` §11.3.

**How it works:** a27_full with one change in where and how the noise enters the generator.
- **Noise only after the 5 decoder blocks** (32, 64, 128, 256, 256 px). Encoder (10 blocks) and bottleneck (3) are noise-free, so the skip connections carry clean features.
- **Per-pixel gate:** noise added = `s_c · sigmoid(conv1×1(x)) · n`. s_c is the learned per-channel strength (init 0.01); the gate is computed from the block's own features (init ≈ 0.5); n is one channel of N(0, 1) per pixel.
- Everything else as a27_full: wavelet-domain discriminator, losses, data v3, 16 + 4 batch, 200k steps.
**4.98 GFLOPs (4.982; the five 1×1 gates add ~0.005), 5.52 M parameters, one pass.**

**Why it was tried:** forcing the noise to matter costs ~0.7 dB even without an adversarial term (a28 phase 1). Models with noise in the encoder path lose 1–4 dB on structured corruptions (`small_artifacts`, `periodic_noise`, `salt_pepper`, blurs), and the same variation is added to near-clean images (`resize_blur` below the input). StyleGAN (Karras et al., CVPR 2019) puts noise only in the synthesis path to create stochastic detail without changing structure. Uncertainty in vision outputs is input-dependent (heteroscedastic; Kendall & Gal, NeurIPS 2017), so the variation should be placed per pixel, not uniformly.
**Is it generative?** Yes: noise input, measured diversity (constraint held at 1.5 grey levels during training), adversarial training from step 0.
**What to look for:** PSNR / SSIM vs a27_full at the same epoch; recovery on structured cases; `resize_blur` and `jpeg` at or above the input; `train_spread255` ~1.5; sample-difference maps look like texture, not blotches.

## Results
Pending. Fill in from `results/a29_gan_decnoise/metrics/`.
