# How to Run: a05_nafnet_v2

Pure-convolution reference: NAFBlocks at every resolution, five scales (256 to 16 px).

For the full server workflow (data preparation, dry run, running every
architecture with one command) see `instructions/README.md`. The commands
below run this architecture alone.

## Rehearse (30 steps; reports speed and peak GPU memory)
```bash
CUDA_VISIBLE_DEVICES=<gpu> python scripts/train.py --arch a05_nafnet_v2 --data_root <data> --dry_run
```

## Train
```bash
CUDA_VISIBLE_DEVICES=<gpu> nohup python scripts/train.py --arch a05_nafnet_v2 --data_root <data> --resume auto > output_a05_nafnet_v2.log 2>&1 &
```
Default: 24 epochs x 2,500 steps = 60,000 steps, batch size 16. `--resume auto`
continues from the last checkpoint if the run was interrupted.

## Evaluate and export
```bash
python scripts/evaluate.py --arch a05_nafnet_v2 --data_root <data>
python scripts/export_inference.py --arch a05_nafnet_v2
```

## Where to look at results
- `results/a05_nafnet_v2/logs/training_log.csv` — one row per epoch
- `results/a05_nafnet_v2/metrics/final_evaluation_summary.json` — the four scored metrics
- `results/a05_nafnet_v2/metrics/benchmark_breakdown.csv` — results per degradation, combination and domain
- `results/a05_nafnet_v2/visualizations/` — corrupted / output / clean samples
- `results/a05_nafnet_v2/submission/` — the exported package

Design notes: `docs/a05_nafnet_v2/notes.md`.
