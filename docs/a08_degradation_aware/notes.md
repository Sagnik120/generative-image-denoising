# a08_degradation_aware — Architecture Notes

a07's body, conditioned on a degradation embedding computed from the input, with a training-only auxiliary task.

## Source
The conditioning idea follows AirNet (Li et al., CVPR 2022) and PromptIR (Potlapalli et al., NeurIPS 2023); the modulation is FiLM (Perez et al., AAAI 2018).

## Why it is in the comparison
One model must handle many corruption types and their combinations. With fixed weights it applies one compromise filter to all of them. Here a small encoder summarises the input's corruption into a 64-d vector (using both the mean and the spread of its features, since noise level is a spread), and that vector rescales and shifts the features of every stage. A linear head must predict from the vector which degradation families were applied and how severely; this label comes free from the data pipeline. The head is never called at inference, so it adds no FLOPs; the encoder adds about 0.05 GFLOPs.

## What to look at in its results
It shares its body with a07, so the a08 minus a07 difference is exactly the value of degradation awareness. Look at the pair and triple groups first. The `aux` value in the log is the auxiliary loss; if it does not fall, the embedding is not learning the corruption.

## Size
About 5 GFLOPs at 256x256 (measured with fvcore; the exact figure is in
`results/a08_degradation_aware/metrics/final_report.json` after a run). All round-2
architectures are held to this budget so that quality differences come from
the design, not from size.

## Knobs (`src/architectures/a08_degradation_aware/config.yaml`)
- `cond_dim`: embedding size.
- `train.aux_weight`: weight of the auxiliary loss (0 disables it).
- `train.loss`: weights of the pixel, frequency, SSIM and DISTS terms.
