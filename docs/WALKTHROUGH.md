# Walkthrough: round 5 (stochastic recipe, a24–a26)

Written 2026-10-14. What was just added, how to get it onto the GPU machine (two 24 GB GPUs), and the commands that run everything. Plan and reasoning: `PLAN.md`. Model details: `architectures/a24_gan_div.md`, `a25_gan_div_wave.md`, `a26_resdiff_gan.md`. Round-4 results: `COMPARISON.md` §9.

## 1. What was completed

Three strictly generative models, ready to train. Same data (pipeline v2), benchmarks, backbone and 60k-step schedule as round 4, all from scratch.

- **a24_gan_div**: a20's conditional GAN + the stochastic recipe. 4.98 GFLOPs.
- **a25_gan_div_wave**: a24 with the wavelet-domain discriminator and wavelet L1 (frequency variant). 4.98 GFLOPs.
- **a26_resdiff_gan**: one-step residual diffusion + GAN. 5.02 GFLOPs.

The stochastic recipe (all three; training-only, no inference FLOPs):
- 8 images per step, each restored with **2 noise draws** (16 generator outputs, same compute as round 4).
- Composite loss on the **mean** of the two samples; 0.2 × Charbonnier on each sample.
- **Diversity hinge**: pushes the two samples ~2 grey levels apart, then stops.
- Adversarial weight 0.01 (round 4: 0.005); noise strength starts at 0.01 (round 4: 0).

Files added or changed:
- `src/common/gan.py`: two-sample training, mean-of-samples loss, per-sample Charbonnier, diversity hinge (`train.stochastic` in the config; off when absent, so a20/a21 are unchanged). New `generate()` hook for subclasses.
- `src/common/gennet.py`: `noise_init` (starting noise strength), test-time `scale`, `set_noise_scale()`.
- `src/common/resdiff.py`: `ResDiffGANBundle` (diffusion + GAN), level shared by the two draws of an image, test-time `noise_scale`.
- `src/common/trainer.py`: new `train_spread255` column in `training_log.csv` (mean difference between the two samples, grey levels). The early warning for collapse.
- `scripts/eval_generative.py`: noise-scale sweep (×0.5, ×1, ×1.5; `--noise_scales`).
- `scripts/run_all.py`: prints the noise scale in its generative table.
- `src/architectures/a24_gan_div/`, `a25_gan_div_wave/`, `a26_resdiff_gan/`: `config.yaml` + `model.py`; `src/registry.py`: three entries.
- Docs: `PLAN.md` (two-GPU schedule), `CURRENT_STATUS.md`, `COMPARISON.md` §9, `architectures/` (a20–a23 results, a24–a26 new, README).

Tested on CPU: 3 training steps per model with all losses (DISTS off, its weights were not available offline), checkpoint save/load, output shape, same seed → identical output, different seed → different output, noise scale 0 → seed-independent output, FLOPs with fvcore (a24 4.978, a26 5.020 GFLOPs). **Not tested on a GPU**: mixed precision, DISTS, memory, speed. The dry run covers these.

## 2. Commit and push (on this machine)

```
git add src scripts docs
git add results/a20_gan results/a21_gan_wavelet results/a22_resdiff results/a23_resdiff_2step results/comparison_round1 results/comparison_round23 results/comparison_round4
git status
git commit -m "feat: add round 5 generative models a24-a26 (stochastic recipe, diffusion+GAN) and round 4 analysis"
git push origin main
```

`git status` also shows deleted files under `results/comparison/` and `results/comparison_old/`. They are left out of the commit; add them with `git add -A results/comparison results/comparison_old` only if the deletion is intended.

## 3. Run on the GPU machine

```
cd <repo on the GPU machine>
git pull origin main
```

**Dry run first** (about 15 minutes; 30 training steps per model, then the real evaluate, generative-eval and export steps; prints speed and peak memory):

```
python scripts/run_all.py --data_root <DATA> --archs a24_gan_div a26_resdiff_gan a25_gan_div_wave --dry_run
```

All three must say `OK`. If one fails, send me `results/_dry_run/<model>/logs/console.log`.

**Real run, one command** (both GPUs, one model per GPU, queue order matters):

```
nohup python scripts/run_all.py --data_root <DATA> --archs a24_gan_div a26_resdiff_gan a25_gan_div_wave > run_round5.log 2>&1 &
```

GPU 0 takes a24, GPU 1 takes a26; whichever GPU frees first takes a25 (in practice GPU 1). For each model: train → evaluate (v1, v2) → generative eval (diversity, noise scales ×0.5/×1/×1.5, and 1/2/4 steps for a26) → export. If interrupted, run the same command again: finished models are skipped, unfinished ones resume.

While it runs:

```
tail -f run_round5.log
tail -f results/a24_gan_div/logs/console.log          # look at spread255 in the step lines
column -s, -t < results/a24_gan_div/logs/training_log.csv | tail -5
nvidia-smi
```

**Collapse check after epoch 2–3 (~1 h):** `spread255` / `train_spread255` should be around 1.5–2.5. If it is below 0.3 and falling, the noise is being switched off again: tell me, and I will raise the diversity weight before you lose a night.

## 4. Time and memory (two 24 GB GPUs)

Times from round 4's measured epoch times (a20 1300 s, a21 1010 s, a22 870 s per 2,500 steps); the new runs do the same generator work per step. Memory is estimated; the dry run gives the measured peak.

| Model | Peak GPU memory (of 24 GB) | 60k steps | Eval + export |
|---|---|---|---|
| a24_gan_div | ~13–16 GB | ~8–9 h | ~25 min (noise sweep) |
| a25_gan_div_wave | ~12–15 GB | ~7 h | ~25 min |
| a26_resdiff_gan | ~14–17 GB | ~7–8 h | ~40 min (steps + noise sweep) |

- **Queue:** GPU 0 a24 (~9 h); GPU 1 a26 (~8 h) then a25 (~7 h). **a24 and a26 done ~9 h after start; a25 ~16 h.**
- **Disk:** ~0.3 GB per model, ~1 GB total.
- **Long run (Day 2 onward):** 200k steps ≈ 29 h for a GAN, ≈ 25 h for a26, one GPU.

If memory is above ~22 GB in the dry run, add `--train_args "--batch_size 6"` (6 images × 2 draws) and tell me.

## 5. What to send back

`run_round5.log` for a first look (comparison and generative tables). For the full analysis, per model:

- `results/<model>/metrics/final_evaluation_summary.json`
- `results/<model>/metrics/generative_eval.json`
- `results/<model>/metrics/final_report.json`
- `results/<model>/logs/training_log.csv`
- `results/<model>/visualizations/epoch_0024_comparison.png`
- `results/comparison/` (CSV files)

First numbers to look at: **diversity** (≥ 0.5 to be eligible, 1–2 expected) and PSNR against a20 (31.89).

## 6. Schedule

| Day | What |
|---|---|
| 1 | commit, pull, dry run, start round 5 (overnight) |
| 2 | morning: send a24/a26 results; I pick the winner and add its 200k config (small code step); **start the long run on GPU 0 by midday**. GPU 1 finishes a25 |
| 3 | long run continues; GPU 1: width-16 sibling or a tuning run (adversarial 0.02) if diversity is low |
| 4 | long run done; evaluation, noise-scale choice; optional v3 corruption fine-tune overnight |
| 5 | final choice, export with the sampler inside the model call, FLOPs (fvcore + ptflops), CPU inference test |
| 6 | report |

Fallback if no round-5 model is eligible: a22 (already trained) with the noise scale that keeps diversity ≥ 0.5.
