# Walkthrough: round 7 (a27_full, a29 on data v3)

Written 2026-10-17. What was just added, how to get it onto the GPU machine (two 24 GB RTX A5000), and the commands that run everything. Plan and reasoning: `PLAN.md`. Model details: `architectures/a27_full.md`, `a29_gan_decnoise.md`. Round-6 results: `COMPARISON.md` §11. Tuning plan for 10-18: `HYPERPARAMETER_TUNING.md`.

## 1. What was completed

Two final 200k-step runs, both on the new **data v3**, ready to train:

- **a27_full**: a27 (round-6 winner) trained 200k steps. Noise after all 18 NAF blocks. 4.978 GFLOPs.
- **a29_gan_decnoise**: a27_full with **noise only in the 5 decoder blocks**, multiplied by a **learned per-pixel gate**. The encoder and bottleneck are noise-free, so structured corruptions are analysed on clean features, and the model can put variation where the image is uncertain and none on near-clean areas. 4.982 GFLOPs.

**Data v3** (both runs):
- v2's 22 corruption families + **4 new ones** aimed at the private set's domains:
  - `demosaic_artifacts`: camera colour fringes and zipper edges (low-light / natural).
  - `chroma_noise`: low-frequency colour blotches (low-light).
  - `mri_ghosting`: shifted faint replicas (MRI).
  - `ct_ring_artifact`: concentric rings (CT).
- 12% clean inputs (v2: 8%), against over-processing of mild corruptions.
- `glass_blur`, `film_grain`, `line_dropout` stay out of training, so benchmark v2 still measures unseen corruptions.
- New **v3 check set**: 4 new families alone and with Gaussian noise, 12 images each (96 images). `evaluate.py` scores it automatically when it has been built.

Unchanged from a27: wavelet-domain discriminator, full composite + wavelet L1 loss on every output, diversity constraint (1.5 grey levels, weight 4.0) on 4 extra draws per batch, adversarial 0.005 from step 0, fresh noise per call in the exported model.

Files added or changed:
- `src/common/degradations.py`: 4 new families, `V3_FAMILIES`, family set `v3`, `BENCHMARK_CASES_V3`.
- `src/common/gennet.py`: `noise_where: decoder` and `noise_gate: true` options (old models build exactly as before).
- `scripts/evaluate.py`: scores the v3 check set (`benchmark_v3` in the summary, `benchmark_v3_breakdown.csv`).
- `scripts/prepare_data.py`: builds the v3 check set with the other benchmarks.
- `scripts/eval_generative.py`: default noise scales 0.5, 0.6, 0.75, 1.0, 1.5.
- `src/architectures/a27_full/`, `a29_gan_decnoise/`: `config.yaml` + `model.py`; `src/registry.py`: two entries.
- Docs: `PLAN.md`, `HYPERPARAMETER_TUNING.md` (new), `CURRENT_STATUS.md`, `COMPARISON.md` §11, `PROJECT_REVIEW_AND_OPTIONS.md` (Options L–N), `architectures/` (round-6 results; a27_full, a29 new; README).

Tested on CPU:
- Each new family at severities 0, 0.5, 1: output shape, range and dtype correct; ~0.2–1.1 ms per image.
- The v3 sampler draws all 4 new families; labels still work.
- a29 has 5 noise modules (a27_full 18); FLOPs with fvcore 4.982 / 4.978 GFLOPs.
- 3 a29 training steps with all losses (DISTS off offline): sample spread ~1.5 from the start, adversarial and discriminator active.
- Same seed gives identical output, different seed or fresh mode gives a different output.
- Round-4–6 models keep exactly the same module structure.

**Not tested on a GPU**: mixed precision, DISTS, memory, speed, the v3 check-set build. The steps below cover these.

## 2. Commit and push (on this machine)

Three commits: round-6 results and analysis, round-7 code, round-7 docs.

```
# 1. round-6 results and analysis
git add results/a25_full results/a27_gan_wave_varctl results/a28_gan_wave_2stage results/comparison_round6
git add docs/COMPARISON.md docs/PROJECT_REVIEW_AND_OPTIONS.md docs/architectures/a25_full.md docs/architectures/a27_gan_wave_varctl.md docs/architectures/a28_gan_wave_2stage.md
git commit -m "docs(results): add round 6 (a25_full, a27, a28) results and analysis; options L-N"

# 2. round-7 code
git add src scripts
git commit -m "feat: add data v3 (4 domain-specific families, 12% clean, v3 check set) and round 7 models a27_full, a29 (decoder-only gated noise)"

# 3. round-7 plan and docs
git add docs
git commit -m "docs: round 7 plan, hyperparameter tuning plan, status, architecture docs and walkthrough"

git status
git push origin main
```

`git status` at the end should show nothing to commit.

## 3. Run on the GPU machine

```
cd <repo on the GPU machine>
git pull origin main
```

**Build the v3 check set once** (~1–2 minutes, no download; uses the same held-out images as benchmarks v1/v2):

```
python -c "from src.common.dataset import build_benchmark; from src.common.degradations import BENCHMARK_CASES_V3; build_benchmark('<DATA>', seed=2028, version='v3', cases=BENCHMARK_CASES_V3)"
```

It should print `building benchmark v3` and end with 96 image pairs in `<DATA>/benchmark/v3/`.

**Dry run** (~15 minutes; 30 training steps per model, then evaluate (now with v3), generative eval and export; prints speed and `gpu_mem`):

```
python scripts/run_all.py --data_root <DATA> --archs a27_full a29_gan_decnoise --dry_run
```

Both must say `OK`.

**Real run, one command** (both GPUs, one model each):

```
nohup python scripts/run_all.py --data_root <DATA> --archs a27_full a29_gan_decnoise > run_round7.log 2>&1 &
```

GPU 0 trains a27_full, GPU 1 trains a29_gan_decnoise. Each: train 200k → evaluate (v1, v2, v3 check set) → generative eval (diversity, noise ×0.5/0.6/0.75/1/1.5) → export. If interrupted, run the same command again: it resumes from the last checkpoint.

While it runs:

```
tail -f run_round7.log
tail -f results/a29_gan_decnoise/logs/console.log     # spread255 and gpu_mem in the step lines
paste -d'\n' <(column -s, -t < results/a27_full/logs/training_log.csv | tail -3) <(column -s, -t < results/a29_gan_decnoise/logs/training_log.csv | tail -3)
nvidia-smi
```

**Checks:**
- **First hour:** `spread255` ~1.5 in both (constraint holds). If a29's falls below 0.5, tell me (the gate is closing).
- **Epoch by epoch:** same schedule, so `val_psnr` of the two runs can be compared at equal epochs. If a29 is more than 0.3 dB behind a27_full after epoch 30, tell me: GPU 1 can then be freed early for tuning.
- **Data workers:** the new families are cheap (~1 ms each); `it/s` should match round 6 (~2.6 it/s). If it is clearly lower, the CPU data loading is the bottleneck: tell me.

## 4. Time and memory (two 24 GB GPUs)

From round 6 (measured: a27 6.9 h for 60k with 20 outputs per step; a25_full 20.4 h for 200k). The gate adds negligible compute. Memory was not printed in the round-6 log; the dry run shows `gpu_mem`.

| Run | GPU | Steps | Time | Peak memory (of 24 GB) |
|---|---|---|---|---|
| v3 check-set build | CPU | – | ~1–2 min | – |
| dry run (both) | 0 + 1 | 30 each | ~15 min | as below |
| a27_full | 0 | 200k | ~21.5 h train + ~0.8 h evaluation | ~14–17 GB |
| a29_gan_decnoise | 1 | 200k | ~21.5–22 h train + ~0.8 h evaluation | ~14–17 GB |

- **Started ~12:00 on 10-17:** both done around **10:00–10:30 on 10-18**.
- **Disk:** ~0.3 GB per model + ~30 MB for the v3 check set.
- If the dry run shows more than ~22 GB: add `--train_args "--batch_size 12"` and tell me.

## 5. What to send back

`run_round7.log` (comparison and generative tables). Per model:

- `results/<model>/metrics/final_evaluation_summary.json` (now includes `benchmark_v3`)
- `results/<model>/metrics/generative_eval.json`
- `results/<model>/metrics/final_report.json`
- `results/<model>/logs/training_log.csv`
- `results/<model>/visualizations/epoch_0080_comparison.png`
- `results/comparison/` (CSV files)

First numbers to look at: PSNR / SSIM / DISTS at each noise scale against a25_full (32.02 / 0.8576 / 0.1399 at ×0.5), diversity ≥ 0.6, and a29 vs a27_full.

## 6. Schedule

| Day | What |
|---|---|
| 10-17 | commit, pull, v3 check set, dry run, start both 200k runs (~12:00). I write the Optuna script during the runs |
| 10-18 | ~10:30 results → architecture choice (`PLAN.md` §5) → Optuna phase A (~11:30 → 10-19 ~05:00) |
| 10-19 | phase B (top-2 + baseline at 30k, ~05:00–12:00) → tuned 200k run (~12:30 → 10-20 ~10:30); export rehearsal of the 10-18 model on GPU 1 |
| 10-20 | evaluate the tuned run, final choice, export (fresh noise per call), fvcore + ptflops, CPU test, report; **submit by 23:59** |

Fallbacks: the 10-18 winner, then a25_full (already trained).
