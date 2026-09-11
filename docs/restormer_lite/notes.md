# restormer_lite — Architecture Notes

## Source
Zamir, S. W., Arora, A., Khan, S., Hayat, M., Khan, F. S., & Yang, M.-H.
(2022). *Restormer: Efficient Transformer for High-Resolution Image
Restoration*. CVPR 2022.

## What problem it solves
Standard Transformer self-attention is quadratic in the number of pixels,
making it far too expensive for high-resolution images. Restormer redesigns
attention to be efficient enough for restoration tasks.

## How it works (plain language)
- A U-Net-shaped encoder-decoder, but each block is a Transformer block
  instead of a plain convolution.
- **Multi-Dconv Head Transposed Attention (MDTA)**: instead of comparing
  every pixel to every other pixel (expensive, quadratic), attention is
  computed **across channels** -- this gives linear complexity relative to
  image size.
- **Gated-Dconv Feed-Forward Network (GDFN)**: the feed-forward block uses
  a gating mechanism (multiply two paths together, one passed through
  GELU) to selectively pass useful information forward, with depthwise
  convolutions mixed in for local spatial context.

## Why try this one
- Historically higher quality ceiling than plain CNNs, especially on
  structure/perceptual metrics (SSIM, DISTS), because self-attention
  captures longer-range dependencies than convolution alone.
- Directly comparable to `nafnet_unet` in this project since both use the
  same U-Net skeleton, downsampling/upsampling, and training loop -- only
  the per-block computation differs. This makes the two an easy head-to-
  head comparison.

## Trade-off to watch
- This is the "heavier" of the two CNN/Transformer options here. Even with
  a reduced `width` (24 vs the original paper's 48) and fewer heads, it
  will typically cost more FLOPs than `nafnet_unet` at matched depth. The
  literature review's Section 5 reports ~1128.9 GFLOPs for full Restormer
  vs ~505.5 GFLOPs for NAFNet at 512x512 -- watch your own measured
  `metrics/final_report.json` GFLOPs number closely against `nafnet_unet`'s.

## Key config knobs (`config.yaml`)
- `width`: base channels (kept at 24, well below the original paper's 48,
  specifically to stay FLOPs-competitive for this task).
- `num_heads`: attention heads per stage -- more heads = finer-grained
  channel grouping, marginally more compute.
- `enc_blocks` / `middle_blocks` / `dec_blocks`: RestormerBlocks per stage.

## Things to try if this underperforms or is too expensive
- Reduce `width` further (e.g. 16) and/or block counts.
- Compare training curves against `nafnet_unet` directly -- if the quality
  gain doesn't justify the extra FLOPs, `nafnet_unet` is likely the better
  submission given the competition's FLOPs-weighted scoring (Section 6 of
  the brief: all four axes are weighted, so a small quality edge isn't
  automatically worth a large FLOPs penalty).
