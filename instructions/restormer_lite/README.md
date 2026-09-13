# How to Run: restormer_lite

Efficient-Transformer alternative to `nafnet_unet` — potentially higher
quality ceiling, higher FLOPs. Directly comparable since both share the
same U-Net skeleton and training loop.

## Train
```bash
python scripts/train.py --arch restormer_lite --data_root /path/to/data
```
Default config: 60 epochs, batch size 8 (smaller than nafnet_unet's 16,
since attention is more memory-hungry), width 24.

If you hit GPU out-of-memory errors on a Colab T4/free-tier GPU, reduce
batch size further:
```bash
python scripts/train.py --arch restormer_lite --data_root /path/to/data --batch_size 4
```

## Evaluate
```bash
python scripts/evaluate.py --arch restormer_lite --data_root /path/to/data
```

## Export submission package
```bash
python scripts/export_inference.py --arch restormer_lite
```

## Where to look at results
Same layout as every other architecture:
`results/restormer_lite/{loss_curves,metrics,visualizations}/...`

## Key thing to watch
Compare `results/restormer_lite/metrics/final_report.json`'s `gflops`
value directly against `results/nafnet_unet/metrics/final_report.json`'s.
If the quality gain (PSNR/SSIM/DISTS) doesn't justify the FLOPs premium,
`nafnet_unet` is likely the stronger submission given the competition's
FLOPs-weighted scoring. See `docs/restormer_lite/notes.md` for more detail.
