# a09_wavelet_naf — Architecture Notes

The NAF U-Net run on Haar wavelet coefficients: 256x256x3 becomes 128x128x12, losslessly.

## Source
NAFNet (ECCV 2022) for the blocks; wavelet-domain restoration follows MWCNN (Liu et al., CVPR Workshops 2018).

## Why it is in the comparison
Every block runs on a quarter of the pixels, so the same FLOPs buy about a third more blocks. The Haar transform is exactly invertible and already separates smooth content from edges and noise.

## What to look at in its results
If a09 matches a05, full-resolution processing is not worth its cost, and the same design at half the depth gives a roughly 2.5 GFLOPs model for the FLOPs axis. If it loses mainly on the low-severity noise cases, fine detail needs the full-resolution level.

## Size
About 5 GFLOPs at 256x256 (measured with fvcore; the exact figure is in
`results/a09_wavelet_naf/metrics/final_report.json` after a run). All round-2
architectures are held to this budget so that quality differences come from
the design, not from size.

## Knobs (`src/architectures/a09_wavelet_naf/config.yaml`)
- `width`: channels at 128 px (40).
- Block counts: each block costs about 0.17 GFLOPs at width 40.
- `train.loss`: weights of the pixel, frequency, SSIM and DISTS terms.
