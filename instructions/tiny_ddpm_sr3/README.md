# How to Run: tiny_ddpm_sr3

Few-step conditional diffusion model. The most experimental of the four
architectures — expect longer training time (80 epochs default, diffusion
objectives converge slower) and higher inference FLOPs (T forward passes,
not 1).

## Train
```bash
python scripts/train.py --arch tiny_ddpm_sr3 --data_root /path/to/data
```
Default config: 80 epochs, batch size 12, `timesteps: 8`.

## Evaluate
```bash
python scripts/evaluate.py --arch tiny_ddpm_sr3 --data_root /path/to/data
```
Note: evaluation runs the FULL T-step sampling loop per validation image
(that's what "inference" means for a diffusion model), so this step is
slower than for the other three architectures — this is expected.

## Export submission package
```bash
python scripts/export_inference.py --arch tiny_ddpm_sr3
```
The exported `inference.py` calls the full T-step `.sample()` method, so
it faithfully reproduces the real inference behavior (and cost) that will
be measured.

## IMPORTANT: reading the FLOPs report correctly
Open `results/tiny_ddpm_sr3/metrics/final_report.json` and look at BOTH:
- `gflops_per_forward_call` — cost of ONE diffusion step
- `true_total_gflops_at_inference` — cost of a FULL inference call
  (`gflops_per_forward_call x timesteps`)

**Compare `true_total_gflops_at_inference` against the other three
architectures' single-pass GFLOPs**, not `gflops_per_forward_call` — the
per-call number alone understates this architecture's real cost.

## If this architecture is too expensive or underperforms
Reduce `timesteps` in `config.yaml` (try 4), or reduce `width` /
block counts to compensate. See `docs/tiny_ddpm_sr3/notes.md` for the
full reasoning and more tuning suggestions.
