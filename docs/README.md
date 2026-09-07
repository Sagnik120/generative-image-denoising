# docs/ — Architecture & Design Notes

This folder holds the *why* behind each architecture's design choices,
pulled from `Generative_Denoising_Literature_Review.md` and adapted
specifically to this competition's constraints (FLOPs-weighted scoring,
zero-shot generalization to undisclosed degradations, cross-domain
robustness to natural/low-light/medical images).

- `nafnet_unet/notes.md` — primary recommendation, lowest FLOPs
- `restormer_lite/notes.md` — efficient Transformer, higher quality ceiling
- `pix2pix_gan/notes.md` — conditional GAN, likely best for perceptual (DISTS) quality
- `tiny_ddpm_sr3/notes.md` — few-step conditional diffusion, most experimental
- `dataset_notes.md` — why DIV2K + BSDS500, and how to add low-light/medical data

See `instructions/` for the practical how-to-run steps for each
architecture; this folder is for the underlying reasoning.
