# a06_restormer_v2 — Architecture Notes

Pure-transformer reference: Restormer blocks (channel attention + gated depthwise feed-forward) at every resolution.

## Source
Zamir et al., *Restormer: Efficient Transformer for High-Resolution Image Restoration*, CVPR 2022.

## Why it is in the comparison
The mirror image of a05: does attention everywhere beat convolution everywhere at equal FLOPs on mixed degradations? A transformer block costs about twice a NAFBlock at the same width, so this network is narrower (width 20) and has fewer blocks.

## What to look at in its results
Round 1 had Restormer and NAFNet 0.12 dB apart, inside that run's measurement noise. The fixed benchmark gives the first real answer.

## Size
About 5 GFLOPs at 256x256 (measured with fvcore; the exact figure is in
`results/a06_restormer_v2/metrics/final_report.json` after a run). All round-2
architectures are held to this budget so that quality differences come from
the design, not from size.

## Knobs (`src/architectures/a06_restormer_v2/config.yaml`)
- `width`, `enc_blocks`, `middle_blocks`, `dec_blocks`: as for a05; a block costs about 0.35 GFLOPs at width 20.
- `ffn_expand`: feed-forward widening (2.0; the paper uses 2.66).
- `num_heads`: one entry per level; each must divide that level's channel count.
- `train.loss`: weights of the pixel, frequency, SSIM and DISTS terms.
