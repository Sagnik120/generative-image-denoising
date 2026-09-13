# How to Run: nafnet_unet

**Recommended starting architecture** — lowest FLOPs, most stable to train.

## Train
```bash
python scripts/train.py --arch nafnet_unet --data_root /path/to/data
```
Default config: 60 epochs, batch size 16, width 32. Edit
`src/architectures/nafnet_unet/config.yaml` to change any of these, or
override on the command line:
```bash
python scripts/train.py --arch nafnet_unet --data_root /path/to/data --epochs 40 --batch_size 24
```

## Evaluate
```bash
python scripts/evaluate.py --arch nafnet_unet --data_root /path/to/data
```

## Export submission package
```bash
python scripts/export_inference.py --arch nafnet_unet
```

## Where to look at results
- `results/nafnet_unet/loss_curves/loss_curve.png` — train/val loss over time
- `results/nafnet_unet/metrics/metrics_dashboard.png` — PSNR/SSIM/DISTS over time
- `results/nafnet_unet/visualizations/epoch_XXXX_comparison.png` — qualitative samples
- `results/nafnet_unet/metrics/final_report.json` — FLOPs, param count, best score

## Quick sanity check before a long run
```bash
python scripts/verify_all.py
```

## If training is unstable or quality plateaus
See `docs/nafnet_unet/notes.md` for specific knobs to try (width,
block counts, `predict_residual`, loss function).
