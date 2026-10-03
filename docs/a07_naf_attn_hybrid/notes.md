# a07_naf_attn_hybrid — Architecture Notes

NAFBlocks at 256/128/64 px, channel attention at 32 px, global spatial self-attention at the 16 px bottleneck.

## Source
NAFNet (ECCV 2022) for the convolutional levels, Restormer (CVPR 2022) for channel attention; the placement follows hybrid designs such as Uformer and X-Restormer.

## Why it is in the comparison
Noise and fine detail are local, and that is where pixels are many and attention is expensive, so those levels use convolution. Large blur kernels and repeated structure need long-range context, and at 16x16 (256 tokens) full self-attention over the whole image costs only tens of MFLOPs per block. The input is fixed at 256x256 in training and testing, so a global operator has no train/test size mismatch.

## What to look at in its results
Compare with a05 on the blur and blur+noise cases: that is where global context should show.

## Size
About 5 GFLOPs at 256x256 (measured with fvcore; the exact figure is in
`results/a07_naf_attn_hybrid/metrics/final_report.json` after a run). All round-2
architectures are held to this budget so that quality differences come from
the design, not from size.

## Knobs (`src/architectures/a07_naf_attn_hybrid/config.yaml`)
- `widths`: channels at each level; the last entry is the bottleneck.
- `enc_blocks`: the last entry is the channel-attention level.
- `middle_blocks`: number of global-attention blocks (about 0.19 GFLOPs each).
- `train.loss`: weights of the pixel, frequency, SSIM and DISTS terms.
