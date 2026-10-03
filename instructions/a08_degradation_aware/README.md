# How to Run: a08_degradation_aware

a07's body, conditioned on a degradation embedding computed from the input, with a training-only auxiliary task.

For the full server workflow (data preparation, dry run, running every
architecture with one command) see `instructions/README.md`. The commands
below run this architecture alone.

## Rehearse (30 steps; reports speed and peak GPU memory)
```bash
CUDA_VISIBLE_DEVICES=<gpu> python scripts/train.py --arch a08_degradation_aware --data_root <data> --dry_run
```

## Train
```bash
CUDA_VISIBLE_DEVICES=<gpu> nohup python scripts/train.py --arch a08_degradation_aware --data_root <data> --resume auto > output_a08_degradation_aware.log 2>&1 &
```
Default: 24 epochs x 2,500 steps = 60,000 steps, batch size 16. `--resume auto`
continues from the last checkpoint if the run was interrupted.

## Evaluate and export
```bash
python scripts/evaluate.py --arch a08_degradation_aware --data_root <data>
python scripts/export_inference.py --arch a08_degradation_aware
```

## Where to look at results
- `results/a08_degradation_aware/logs/training_log.csv` — one row per epoch
- `results/a08_degradation_aware/metrics/final_evaluation_summary.json` — the four scored metrics
- `results/a08_degradation_aware/metrics/benchmark_breakdown.csv` — results per degradation, combination and domain
- `results/a08_degradation_aware/visualizations/` — corrupted / output / clean samples
- `results/a08_degradation_aware/submission/` — the exported package

Design notes: `docs/a08_degradation_aware/notes.md`.
