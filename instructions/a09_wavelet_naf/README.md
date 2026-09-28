# How to Run: a09_wavelet_naf

The NAF U-Net run on Haar wavelet coefficients: 256x256x3 becomes 128x128x12, losslessly.

For the full server workflow (data preparation, dry run, running every
architecture with one command) see `instructions/README.md`. The commands
below run this architecture alone.

## Rehearse (30 steps; reports speed and peak GPU memory)
```bash
CUDA_VISIBLE_DEVICES=<gpu> python scripts/train.py --arch a09_wavelet_naf --data_root <data> --dry_run
```

## Train
```bash
CUDA_VISIBLE_DEVICES=<gpu> nohup python scripts/train.py --arch a09_wavelet_naf --data_root <data> --resume auto > output_a09_wavelet_naf.log 2>&1 &
```
Default: 24 epochs x 2,500 steps = 60,000 steps, batch size 16. `--resume auto`
continues from the last checkpoint if the run was interrupted.

## Evaluate and export
```bash
python scripts/evaluate.py --arch a09_wavelet_naf --data_root <data>
python scripts/export_inference.py --arch a09_wavelet_naf
```

## Where to look at results
- `results/a09_wavelet_naf/logs/training_log.csv` — one row per epoch
- `results/a09_wavelet_naf/metrics/final_evaluation_summary.json` — the four scored metrics
- `results/a09_wavelet_naf/metrics/benchmark_breakdown.csv` — results per degradation, combination and domain
- `results/a09_wavelet_naf/visualizations/` — corrupted / output / clean samples
- `results/a09_wavelet_naf/submission/` — the exported package

Design notes: `docs/a09_wavelet_naf/notes.md`.
