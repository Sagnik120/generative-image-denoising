# a11_freq_spatial — Architecture Notes

NAFBlocks at full resolution; below that, blocks with a spatial path and a Fourier-domain path in parallel.

## Source
Mao et al., *Intriguing Findings of Frequency Selection for Image Deblurring (DeepRFT)*, AAAI 2023; Kong et al., *FFTformer*, CVPR 2023.

## Why it is in the comparison
Blur is a convolution, which is a per-frequency multiplication in the Fourier domain; JPEG blocking, stripes and periodic patterns are also compact in frequency. One FFT gives a block a global view that stacked 3x3 convolutions need many layers to reach. Noise is handled by the spatial path. The FFT itself has no learned weights and is not counted by fvcore or ptflops; the 1x1 convolutions applied to the spectrum are.

## What to look at in its results
Compare with a05 (same network without the frequency path) on blur, JPEG and the blur+noise+JPEG triples.

## Size
About 5 GFLOPs at 256x256 (measured with fvcore; the exact figure is in
`results/a11_freq_spatial/metrics/final_report.json` after a run). All round-2
architectures are held to this budget so that quality differences come from
the design, not from size.

## Knobs (`src/architectures/a11_freq_spatial/config.yaml`)
- `freq_reduction`: channel reduction inside the Fourier branch (2). 4 makes each block about 15% cheaper.
- Block counts: a dual-domain block costs about 0.31 GFLOPs at width 24.
- `train.loss`: weights of the pixel, frequency, SSIM and DISTS terms.
