# docs/ — Architecture & Design Notes

This folder holds the *why* behind each architecture's design choices,
pulled from `Generative_Denoising_Literature_Review.md` and adapted
specifically to this competition's constraints (FLOPs-weighted scoring,
zero-shot generalization to undisclosed degradations, cross-domain
robustness to natural/low-light/medical images).

Round 2 (new pipeline, about 5 GFLOPs each, scored on the fixed benchmark):
- `a05_nafnet_v2/notes.md` — pure-convolution reference
- `a06_restormer_v2/notes.md` — pure-transformer reference
- `a07_naf_attn_hybrid/notes.md` — convolution at high resolution, global attention at the bottleneck
- `a08_degradation_aware/notes.md` — a07 conditioned on a learned degradation embedding
- `a09_wavelet_naf/notes.md` — NAF U-Net in the Haar wavelet domain
- `a10_resflow_fewstep/notes.md` — generative residual flow, two passes
- `a11_freq_spatial/notes.md` — spatial and Fourier-domain paths in every block

Round 1 (original DIV2K-only pipeline; kept for reference, not comparable):
- `a01_nafnet_unet/notes.md`, `a02_restormer_lite/notes.md`,
  `a03_pix2pix_gan/notes.md`, `a04_tiny_ddpm_sr3/notes.md`

Shared:
- `dataset_notes.md` — the datasets, how they are mixed, the degradation
  pipeline and the fixed benchmark

See `instructions/` for the practical how-to-run steps for each
architecture; this folder is for the underlying reasoning.
