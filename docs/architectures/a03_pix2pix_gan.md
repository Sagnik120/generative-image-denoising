# a03 Pix2Pix GAN

**How it works:** a01 generator + PatchGAN discriminator; L1 (λ=100) + adversarial loss. Aims for sharper textures.
**Observation:** GAN did not help pixel/perceptual metrics here.

## Results
Round 1 (old pipeline, not comparable with round 2): PSNR 26.54, SSIM 0.7200, DISTS 0.2460, 12.82 GFLOPs.
