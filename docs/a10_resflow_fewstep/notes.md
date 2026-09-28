# a10_resflow_fewstep — Architecture Notes

Generative residual flow from the corrupted image to the clean one, run in 2 passes.

## Source
Delbracio & Milanfar, *Inversion by Direct Iteration (InDI)*, TMLR 2023; Liu et al., *Rectified Flow*, ICLR 2023.

## Why it is in the comparison
a04 diffused from pure noise and collapsed. This model starts from the corrupted image: it is trained to predict the clean image from any point on the straight line between the two, and inference walks that line in K steps, each refining the last. Half of the training batches use the model's own first prediction as the starting point, so the second pass is trained on what it will actually receive.

## What to look at in its results
All K passes run inside one call, so the measured FLOPs are the true total; each pass therefore gets about half of the budget. Expect DISTS to be its strong axis and PSNR its weak one. Set `steps: 1` in the config to evaluate the same weights as a single pass.

## Size
About 5 GFLOPs at 256x256 (measured with fvcore; the exact figure is in
`results/a10_resflow_fewstep/metrics/final_report.json` after a run). All round-2
architectures are held to this budget so that quality differences come from
the design, not from size.

## Knobs (`src/architectures/a10_resflow_fewstep/config.yaml`)
- `steps`: passes per call. FLOPs scale with it.
- `train.self_pred_prob`: share of batches trained from the model's own first-step output.
- `train.loss`: weights of the pixel, frequency, SSIM and DISTS terms.
