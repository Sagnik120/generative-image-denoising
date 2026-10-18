# Walkthrough: round 8 final runs (Optuna trials 32 and 37, 80 epochs)

Written 2026-10-19. What was just added, and the commands to commit, push and start both final runs on the GPU machine (two 24 GB RTX A5000). Plan: `PLAN.md` §6. Tuning results and the choice of trials: `HYPERPARAMETER_TUNING.md` §5.

## 1. What was completed

**Two final runs, ready to start with one `run_all.py` command:**

| Run | GPU | Optuna trial | lr | D/G lr ratio | adversarial | diversity target | SSIM weight | Steps |
|---|---|---|---|---|---|---|---|---|
| `a29_tuned` | 0 | 32 (safest of the top trials) | 7.55e-4 | 0.5 | 0.00275 | 1.60 | 0.334 | 200k (80 epochs) |
| `a29_tuned_b` | 1 | 37 (best J) | 7.91e-4 | 2.0 | 0.00334 | 1.37 | 0.189 | 200k (80 epochs) |

Everything else is exactly a29: decoder-only gated noise (4.98 GFLOPs), wavelet-domain discriminator, composite + wavelet loss, diversity constraint, data v3, batch 16 + 4 draws, EMA 0.999, 1k warm-up and cosine schedule. **No early stopping:** both train all 80 epochs. After training, `run_all.py` evaluates them (v1, v2, v3), runs the noise-scale table (×0.5 / 0.6 / 0.75 / 1 / 1.5) and exports, as in every earlier round.

**Code changes:**
- `scripts/tune_optuna.py`: `--write_configs --trial N --arch <name> [--epochs 80]` writes the config of any chosen trial, not only the best. Used here to write trial 32 → `a29_tuned` and trial 37 → `a29_tuned_b` from the study database (`results/optuna/a29_tune.db`, the same file as on `/DATA1`). **Both configs are already written with the tuned values**: nothing to generate on the GPU machine.
- `src/architectures/a29_tuned_b/` (new) and `src/registry.py`: entry for trial 37.
- `scripts/evaluate.py`: `--noise_scale S` evaluates a generative model at test-time noise ×S. Any value other than 1 writes `final_evaluation_summary_noise<S>.json` and `benchmark*_breakdown_noise<S>.csv` (v1 groups/domains, v2, v3), so the default files are kept. Used on 10-20 for the report breakdown at the chosen scale.
- `scripts/eval_generative.py`: `--out_name` writes extra noise scales (e.g. ×0.4) to a separate file instead of overwriting `generative_eval.json`.

**Tested on CPU:** both configs written from the real study and checked by building the models (lr, D/G ratio, adversarial weight, diversity target, SSIM weight, 200k steps, data v3 all as in the table). Both evaluation scripts parse. **Not tested on a GPU:** the dry run in §3 covers it.

## 2. Commit and push (on this machine)

```
# 1. Optuna results and round-8 docs
git add results/optuna
git add docs
git commit -m "docs(results): Optuna search on a29 (38 trials); trials 32 and 37 chosen for final 80-epoch runs"

# 2. round-8 final-run code and configs
git add scripts/tune_optuna.py scripts/evaluate.py scripts/eval_generative.py src
git commit -m "feat: configs for Optuna trials 32 (a29_tuned) and 37 (a29_tuned_b); --noise_scale in evaluate, --out_name in eval_generative"

git status
git push origin main
```

## 3. Run on the GPU machine

```
cd <repo on the GPU machine>
git pull origin main
export R=/DATA1/shrusti/genai_project/results
head -2 src/architectures/a29_tuned/config.yaml src/architectures/a29_tuned_b/config.yaml   # trial 32 / trial 37 values
```

**Dry run (~15 min):**

```
python scripts/run_all.py --data_root <DATA> --results_root $R --archs a29_tuned a29_tuned_b --dry_run
```

Both must say `OK`.

**Real run, one command (both GPUs, one model each, full 80 epochs):**

```
nohup python scripts/run_all.py --data_root <DATA> --results_root $R --archs a29_tuned a29_tuned_b > $R/run_round8_final.log 2>&1 &
```

While it runs:

```
tail -f $R/run_round8_final.log
tail -f $R/a29_tuned/logs/console.log           # spread255, gpu_mem
column -s, -t < $R/a29_tuned/logs/training_log.csv | tail -3
column -s, -t < $R/a29_tuned_b/logs/training_log.csv | tail -3
nvidia-smi
```

For reference, a29 at the same epochs (val PSNR at noise ×1): epoch 12: 30.97, 24: 31.66, 40: 32.08, 80: 31.93; DISTS at 80: 0.135; spread 1.65–2.1. If interrupted, run the same command again: it resumes from the last checkpoint.

## 4. On 10-20, after the runs (~30 min of extras)

1. **Choose** from the noise-scale tables printed at the end of `run_round8_final.log` (a29's is in `$R/a29_gan_decnoise/metrics/generative_eval.json`). Rules: `PLAN.md` §6.4.
2. **Noise ×0.4** (only for a model with diversity ≥ 0.75 at ×0.5; ~12 min each, both GPUs in parallel; the model's ×1 is scored again in the same call):
   ```
   CUDA_VISIBLE_DEVICES=0 python scripts/eval_generative.py --arch a29_tuned --data_root <DATA> --results_root $R --noise_scales 0.4 --out_name generative_eval_x04.json &
   CUDA_VISIBLE_DEVICES=2 python scripts/eval_generative.py --arch a29_tuned_b --data_root <DATA> --results_root $R --noise_scales 0.4 --out_name generative_eval_x04.json &
   ```
3. **Breakdown at the chosen scale for the final model** (~20 min; replace `<FINAL>` and `<S>`, e.g. `a29_tuned` and `0.5`):
   ```
   python scripts/evaluate.py --arch <FINAL> --data_root <DATA> --results_root $R --noise_scale <S>
   ```
   Writes `$R/<FINAL>/metrics/final_evaluation_summary_noise<S>.json` and the `_noise<S>` breakdown CSVs (v1 groups and domains, v2, v3).

## 5. Time and memory (two 24 GB GPUs)

| Step | GPU | Time | Peak memory (of 24 GB) |
|---|---|---|---|
| dry run (both) | 0 + 1 | ~15 min | as below |
| `a29_tuned`, 80 epochs + evaluation + noise table + export | 0 | ~25–26 h (as a29: 25.3 h at the current machine speed; ~21 h at normal speed) | ~14–17 GB |
| `a29_tuned_b`, same | 1 | ~25–26 h | ~14–17 GB |
| 10-20: noise ×0.4 per model | 0 / 1 | ~12 min | ~4–6 GB |
| 10-20: breakdown at the chosen scale | 0 | ~20 min | ~4–6 GB |

- **Started ~11:00 on 10-19:** both done ~12:00–13:00 on 10-20.
- **Disk:** ~0.3 GB per run on `$R`.

## 6. Schedule and what to send back

| When | What |
|---|---|
| 10-19 now | commit, push, pull, dry run, **start both runs** |
| 10-20 ~12:00–13:00 | runs done → send `$R/run_round8_final.log` and, for each run: `metrics/final_evaluation_summary.json`, `metrics/generative_eval.json`, `logs/training_log.csv`, `visualizations/epoch_0080_comparison.png` |
| 10-20 ~13:00–13:30 | final choice among a29 / a29_tuned / a29_tuned_b; noise ×0.4 and the breakdown at the chosen scale |
| 10-20 afternoon | export of the final model (fresh noise per call), CPU test, report; **submit by 23:59** |

Fallbacks: a29 (trained), then a25_full.
