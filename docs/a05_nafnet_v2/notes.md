# a05_nafnet_v2 — Architecture Notes

Pure-convolution reference: NAFBlocks at every resolution, five scales (256 to 16 px).

## Source
Chen, Chu, Zhang & Sun, *Simple Baselines for Image Restoration (NAFNet)*, ECCV 2022.

## Why it is in the comparison
It answers "how far does a plain CNN get under the new data, degradations and losses?". Every other round-2 architecture is read against it at the same FLOPs. It differs from a01 in size and shape only: width 24 instead of 32, one extra scale, and most blocks in the coarse encoder levels, bringing it from 12.8 to about 5 GFLOPs.

## What to look at in its results
Its numbers are the baseline. A candidate that does not beat a05 on the benchmark is not worth its extra complexity.

## Size
About 5 GFLOPs at 256x256 (measured with fvcore; the exact figure is in
`results/a05_nafnet_v2/metrics/final_report.json` after a run). All round-2
architectures are held to this budget so that quality differences come from
the design, not from size.

## Knobs (`src/architectures/a05_nafnet_v2/config.yaml`)
- `width`: cost grows with its square. 16 gives roughly 2.3 GFLOPs, 32 roughly 8.7.
- `enc_blocks` / `middle_blocks` / `dec_blocks`: each block costs about 0.24 GFLOPs at width 24, at any level.
- `train.loss`: weights of the pixel, frequency, SSIM and DISTS terms.
