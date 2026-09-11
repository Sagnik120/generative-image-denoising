# pix2pix_gan — Architecture Notes

## Source
Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). *Image-to-Image
Translation with Conditional Adversarial Networks*. CVPR 2017.

## What problem it solves
Plain pixel-wise losses (L1/L2) reward the "safe average" answer, which
tends to produce blurry outputs -- especially visible on fine texture and
edges. Adversarial training pushes the network toward *realistic* looking
outputs instead of merely low-error ones.

## How it works (plain language)
- A **generator** (the same NAFBlock-based U-Net used in `nafnet_unet`,
  reused here so the two architectures are directly comparable) maps
  corrupted -> restored images.
- A **PatchGAN discriminator** looks at 70x70 overlapping patches of
  (corrupted, output) vs (corrupted, real-clean) pairs and tries to tell
  them apart.
- The generator is trained with a weighted combination of:
  - Adversarial loss (fool the discriminator)
  - Reconstruction loss (L1 by default, weighted heavily via
    `lambda_recon: 100.0`) so it still stays close to the ground truth,
    not just "realistic-looking but wrong."

## Why try this one
- Likely to help the **DISTS** (perceptual) score specifically, since
  DISTS is designed to reward realistic texture/structure over pure pixel
  accuracy -- exactly what adversarial training optimizes for.
- The discriminator is **training-only** -- it is thrown away for the
  final submission, so this approach costs you nothing extra at
  inference/FLOPs time. Only the generator (identical architecture to
  `nafnet_unet`) ships.

## Trade-off to watch
- GAN training is less stable than plain supervised regression: watch the
  `gan_loss_curve.png` (generator vs discriminator loss) for signs of
  mode collapse (discriminator loss collapsing to near-zero while
  generator loss climbs) or oscillation.
- PSNR/SSIM may come in slightly *lower* than the plain `nafnet_unet`
  (which optimizes pixel accuracy directly), even if DISTS improves --
  this is the classic "realism vs pixel-fidelity" trade-off the
  literature review's Pix2Pix section describes.

## Key config knobs (`config.yaml`)
- `lambda_recon`: weight of the reconstruction loss relative to the
  adversarial loss. Higher = more like plain supervised training (safer,
  less "GAN-y"); lower = more aggressive realism push (riskier, can
  hallucinate detail that isn't really there -- watch out for
  hallucinated texture hurting PSNR on the held-out set).
- `disc_layers` / `disc_base_ch`: discriminator capacity. Usually doesn't
  need tuning much.

## Things to try if this underperforms
- Increase `lambda_recon` if outputs look "hallucinated" or PSNR drops
  too much relative to `nafnet_unet`.
- Decrease it if outputs still look blurry / GAN isn't having any visible
  effect on texture sharpness.
- Compare the `visualizations/epoch_*_comparison.png` grids side-by-side
  with `nafnet_unet`'s at the same epoch to see the realism difference
  directly.
