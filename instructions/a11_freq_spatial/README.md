# How to Run: a11_freq_spatial

NAFBlocks at full resolution; below that, blocks with a spatial path and a Fourier-domain path in parallel.

For the full server workflow (data preparation, dry run, running every
architecture with one command) see `instructions/README.md`. The commands
below run this architecture alone.

## Rehearse (30 steps; reports speed and peak GPU memory)
```bash
CUDA_VISIBLE_DEVICES=<gpu> python scripts/train.py --arch a11_freq_spatial --data_root <data> --dry_run
```

## Train
```bash
CUDA_VISIBLE_DEVICES=<gpu> nohup python scripts/train.py --arch a11_freq_spatial --data_root <data> --resume auto > output_a11_freq_spatial.log 2>&1 &
```
Default: 24 epochs x 2,500 steps = 60,000 steps, batch size 16. `--resume auto`
continues from the last checkpoint if the run was interrupted.

## Evaluate and export
```bash
python scripts/evaluate.py --arch a11_freq_spatial --data_root <data>
python scripts/export_inference.py --arch a11_freq_spatial
```

## Where to look at results
- `results/a11_freq_spatial/logs/training_log.csv` — one row per epoch
- `results/a11_freq_spatial/metrics/final_evaluation_summary.json` — the four scored metrics
- `results/a11_freq_spatial/metrics/benchmark_breakdown.csv` — results per degradation, combination and domain
- `results/a11_freq_spatial/visualizations/` — corrupted / output / clean samples
- `results/a11_freq_spatial/submission/` — the exported package

Design notes: `docs/a11_freq_spatial/notes.md`.
