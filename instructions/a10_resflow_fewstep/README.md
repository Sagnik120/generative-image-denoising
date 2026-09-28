# How to Run: a10_resflow_fewstep

Generative residual flow from the corrupted image to the clean one, run in 2 passes.

For the full server workflow (data preparation, dry run, running every
architecture with one command) see `instructions/README.md`. The commands
below run this architecture alone.

## Rehearse (30 steps; reports speed and peak GPU memory)
```bash
CUDA_VISIBLE_DEVICES=<gpu> python scripts/train.py --arch a10_resflow_fewstep --data_root <data> --dry_run
```

## Train
```bash
CUDA_VISIBLE_DEVICES=<gpu> nohup python scripts/train.py --arch a10_resflow_fewstep --data_root <data> --resume auto > output_a10_resflow_fewstep.log 2>&1 &
```
Default: 24 epochs x 2,500 steps = 60,000 steps, batch size 16. `--resume auto`
continues from the last checkpoint if the run was interrupted.

## Evaluate and export
```bash
python scripts/evaluate.py --arch a10_resflow_fewstep --data_root <data>
python scripts/export_inference.py --arch a10_resflow_fewstep
```

## Where to look at results
- `results/a10_resflow_fewstep/logs/training_log.csv` — one row per epoch
- `results/a10_resflow_fewstep/metrics/final_evaluation_summary.json` — the four scored metrics
- `results/a10_resflow_fewstep/metrics/benchmark_breakdown.csv` — results per degradation, combination and domain
- `results/a10_resflow_fewstep/visualizations/` — corrupted / output / clean samples
- `results/a10_resflow_fewstep/submission/` — the exported package

Design notes: `docs/a10_resflow_fewstep/notes.md`.
