# How to Run: a07_naf_attn_hybrid

NAFBlocks at 256/128/64 px, channel attention at 32 px, global spatial self-attention at the 16 px bottleneck.

For the full server workflow (data preparation, dry run, running every
architecture with one command) see `instructions/README.md`. The commands
below run this architecture alone.

## Rehearse (30 steps; reports speed and peak GPU memory)
```bash
CUDA_VISIBLE_DEVICES=<gpu> python scripts/train.py --arch a07_naf_attn_hybrid --data_root <data> --dry_run
```

## Train
```bash
CUDA_VISIBLE_DEVICES=<gpu> nohup python scripts/train.py --arch a07_naf_attn_hybrid --data_root <data> --resume auto > output_a07_naf_attn_hybrid.log 2>&1 &
```
Default: 24 epochs x 2,500 steps = 60,000 steps, batch size 16. `--resume auto`
continues from the last checkpoint if the run was interrupted.

## Evaluate and export
```bash
python scripts/evaluate.py --arch a07_naf_attn_hybrid --data_root <data>
python scripts/export_inference.py --arch a07_naf_attn_hybrid
```

## Where to look at results
- `results/a07_naf_attn_hybrid/logs/training_log.csv` — one row per epoch
- `results/a07_naf_attn_hybrid/metrics/final_evaluation_summary.json` — the four scored metrics
- `results/a07_naf_attn_hybrid/metrics/benchmark_breakdown.csv` — results per degradation, combination and domain
- `results/a07_naf_attn_hybrid/visualizations/` — corrupted / output / clean samples
- `results/a07_naf_attn_hybrid/submission/` — the exported package

Design notes: `docs/a07_naf_attn_hybrid/notes.md`.
