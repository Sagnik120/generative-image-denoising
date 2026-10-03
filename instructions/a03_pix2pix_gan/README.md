# How to Run: a03_pix2pix_gan

Conditional GAN — same generator architecture as `a01_nafnet_unet`, plus a
PatchGAN discriminator used only during training (thrown away for the
final submission, so no extra inference FLOPs cost).

## Train
```bash
python scripts/train.py --arch a03_pix2pix_gan --data_root /path/to/data
```
Default config: 60 epochs, batch size 16, `lambda_recon=100.0`.

## Evaluate
```bash
python scripts/evaluate.py --arch a03_pix2pix_gan --data_root /path/to/data
```
Note: only the **generator** is evaluated (matches what would actually
ship in a submission).

## Export submission package
```bash
python scripts/export_inference.py --arch a03_pix2pix_gan
```
This correctly extracts and packages ONLY the generator weights — the
discriminator is not included (it isn't needed for inference).

## Where to look at results
- `results/a03_pix2pix_gan/loss_curves/gan_loss_curve.png` — generator vs
  discriminator loss (watch for instability: a discriminator loss that
  collapses toward zero while generator loss climbs indicates the
  generator has stopped being able to fool it — consider lowering the
  discriminator's learning rate or adjusting `lambda_recon`).
- `results/a03_pix2pix_gan/loss_curves/loss_curve.png` — the reconstruction-
  loss-only curve (this is what's directly comparable to a01_nafnet_unet's
  loss curve).
- `results/a03_pix2pix_gan/visualizations/epoch_XXXX_comparison.png` —
  compare texture sharpness against a01_nafnet_unet's equivalent-epoch grid.

## If GAN training looks unstable
See `docs/a03_pix2pix_gan/notes.md` — usually a `lambda_recon` adjustment
(try 50 or 200) resolves it.
