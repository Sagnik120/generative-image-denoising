# Walkthrough: round 8 (hyperparameter tuning of a29)

Written 2026-10-18. What was just added, how to get it onto the GPU machine (two 24 GB RTX A5000), and the commands for the three steps: Optuna search → 30k confirmation → final 80-epoch run. Plan and rules: `HYPERPARAMETER_TUNING.md`, `PLAN.md` §6. Round-7 results: `COMPARISON.md` §12.

## 1. What was completed

**Final architecture:** a29_gan_decnoise (32.27 / 0.8682 / 0.1360 at noise ×0.5, diversity 0.86). Round 8 tunes five of its hyperparameters and retrains it.

**Is 80 epochs enough?** Checked on a29's own curve:
- over the last 12 epochs PSNR moved +0.02 dB and DISTS −0.001, and the training loss was flat (~0.049) with the learning rate at ~1e-6;
- PSNR peaked at epoch 44, and the second half traded a little PSNR for better DISTS.

The schedule reaches its plateau, so 80 epochs is not stopping the model early. A 100-epoch schedule would add at most ~0.05–0.15 dB and would end too close to the deadline.

**Code added:**
- **`scripts/tune_optuna.py`** (new). Optuna search over a29:
  - **Search space (5):** learning rate 2e-4–8e-4 (log); discriminator/generator lr ratio {0.5, 1, 2}; adversarial weight 0.0025–0.01 (log); diversity target 1.0–1.8 grey levels; SSIM weight 0.1–0.35.
  - **Trials:** cap **40 trials in total** (both workers together) or **14 h per worker**. Each trial is **10k steps = 4 epochs** with its own warm-up (500 steps) and cosine schedule; the adversarial ramp is shortened to 1k steps.
  - **Sampler:** TPE (6 random start-up trials). **Trial 0 = the current a29 setting.**
  - **Pruner:** successive halving; checks at 2.5k and 5k steps, so weak trials stop after ~20–40 minutes.
  - **Score:** `J = PSNR + 20·SSIM − 20·DISTS` on a 132-image subset of benchmark v1 (4 per case: 2 natural, 1 low-light, 1 medical), **at noise ×0.5**.
  - **Fails a trial if:** diversity < 0.6 (16 images × 4 seeds), final sample spread > 3× the diversity target (a27_full's failure), or the losses keep going non-finite.
  - **Outputs (on the HDD, not in the repo):** an SQLite study shared by the two workers (`/DATA1/shrusti/genai_project/results/optuna/a29_tune.db`) and one row per trial in `/DATA1/shrusti/genai_project/results/optuna/trials_a29_tune.csv`. The results root defaults to `/DATA1/shrusti/genai_project/results` when that folder exists; `--results_root` or the `GENAI_RESULTS_ROOT` variable override it.
  - `--summary` prints the top 10 trials. `--write_configs` writes the phase-B and final configs from the best trial.
- `src/common/gan.py`: `gan.disc_lr_ratio` (discriminator learning rate = ratio × generator learning rate; default 1.0, so every earlier model is unchanged).
- New configs, all a29's network:
  - `a29_base_30k`: the a29 setting at 30k steps, the phase-B baseline.
  - `a29_tuned_30k`: the best setting at 30k steps.
  - `a29_tuned`: the best setting at 200k steps (80 epochs).

  The two `tuned` configs are **placeholders equal to a29 until `--write_configs` overwrites them.**
- `src/registry.py`: three entries. `requirements.txt`: `optuna`.
- Docs: `HYPERPARAMETER_TUNING.md`, `PLAN.md`, `CURRENT_STATUS.md`, `COMPARISON.md` §12, `architectures/` (a27_full and a29 results, round-8 table).

**Tested on CPU with stand-in data:**
- two complete mini-trials: trial 0 = baseline, trial 1 from TPE;
- scores reported at the intermediate checkpoints;
- trial log and summary written;
- `--write_configs` produced valid configs from the best trial (placeholders restored afterwards);
- the 10k-step trial schedule checked (4 epochs × 2.5k, warm-up 500, ramp 1k).

**Not tested on a GPU** (speed, memory, real data): the smoke run in §3 covers this.

## 2. Commit and push (on this machine)

```
# 1. round-7 results and analysis
git add results/a27_full results/a29_gan_decnoise results/comparison_round7
git add docs/COMPARISON.md docs/CURRENT_STATUS.md docs/architectures/a27_full.md docs/architectures/a29_gan_decnoise.md
git commit -m "docs(results): add round 7 (a27_full, a29) results; a29 chosen as final architecture"

# 2. round-8 tuning code
git add scripts/tune_optuna.py src requirements.txt
git commit -m "feat: add Optuna tuning of a29 (scripts/tune_optuna.py), disc lr ratio, round 8 configs"

# 3. round-8 docs
git add docs
git commit -m "docs: hyperparameter tuning plan, round 8 plan and walkthrough"

git status
git push origin main
```

## 3. Run on the GPU machine

```
cd <repo on the GPU machine>
git pull origin main
pip install "optuna>=3.4"
export R=/DATA1/shrusti/genai_project/results          # all results stay on the HDD
mkdir -p $R/optuna
```

Every command below writes to `$R`: `tune_optuna.py` by default (and with `--results_root $R` to be explicit), and `run_all.py` through `--results_root $R`. Nothing new is created inside the code folder.

The run log of round 7 shows our GPUs as **0 and 2**; check with `nvidia-smi` and use the free ones below.

**Step 0: smoke test (~5–10 min, separate study, nothing kept).** 2 tiny trials of 60 steps on real data:

```
CUDA_VISIBLE_DEVICES=0 python scripts/tune_optuna.py --data_root <DATA> --results_root $R --smoke --n_trials 2 --enqueue_baseline --study_name smoke
```

It must print `step 20` / `step 40` lines with J, PSNR, SSIM, DISTS, diversity for 2 trials, then a summary. Then: `rm $R/optuna/smoke.db $R/optuna/trials_smoke.csv`.

**Step 1: Optuna search (phase A, ~14 h), two workers on one study.** Start worker 1 a minute after worker 0 (worker 0 queues the baseline as trial 0):

```
CUDA_VISIBLE_DEVICES=0 nohup python scripts/tune_optuna.py --data_root <DATA> --results_root $R --enqueue_baseline > $R/optuna/tune_w0.log 2>&1 &
sleep 60
CUDA_VISIBLE_DEVICES=2 nohup python scripts/tune_optuna.py --data_root <DATA> --results_root $R > $R/optuna/tune_w1.log 2>&1 &
```

While it runs:

```
tail -f $R/optuna/tune_w0.log $R/optuna/tune_w1.log
column -s, -t < $R/optuna/trials_a29_tune.csv
python scripts/tune_optuna.py --results_root $R --summary
nvidia-smi
```

Both workers stop by themselves after 40 trials in total or 14 h. Check: trial 0 (baseline) should reach J ≈ PSNR + 20·SSIM − 20·DISTS of a short a29 run; diversity values should be ≥ 0.6 for most trials.

**Step 2: write the configs from the best trial, then confirm at 30k (phase B, ~4.5 h).**

```
python scripts/tune_optuna.py --results_root $R --summary
python scripts/tune_optuna.py --results_root $R --write_configs
head -2 src/architectures/a29_tuned/config.yaml        # shows the tuned values
nohup python scripts/run_all.py --data_root <DATA> --results_root $R --archs a29_tuned_30k a29_base_30k > $R/run_round8_phaseB.log 2>&1 &
```

Send me `$R/run_round8_phaseB.log` and the `--summary` output. **Decision (by me, same hour):** the final run uses the tuned setting only if `a29_tuned_30k` beats `a29_base_30k` by ≥ 0.1 in J at noise ×0.5, with diversity ≥ 0.6 and a stable spread. Otherwise no final run: a29 is final.

**Step 3: final run, 80 epochs (~21–27 h).**

```
nohup python scripts/run_all.py --data_root <DATA> --results_root $R --archs a29_tuned > $R/run_round8_final.log 2>&1 &
```

It trains, evaluates (v1, v2, v3), runs the generative evaluation (noise ×0.5–1.5) and exports, like every earlier run. The second GPU is free during this run.

## 4. Time and memory (two 24 GB GPUs)

At the current machine speed (~1220 s per 2.5k steps).

| Step | GPU | Time | Peak memory (of 24 GB) |
|---|---|---|---|
| 0 smoke test | 1 | ~5–10 min | ~14–17 GB |
| 1 Optuna, per complete trial (10k steps + 3 subset evaluations) | each | ~1.45 h; pruned trials ~0.4–0.8 h | ~14–17 GB per worker (one trial per GPU) |
| 1 Optuna, total | both | ~14 h (cap 40 trials; ~8–12 complete) | |
| 2 phase B (two 30k runs in parallel) | both | ~4.3 h + ~0.5 h evaluation | ~14–17 GB each |
| 3 final `a29_tuned`, 80 epochs | 1 | ~27 h (~21 h at normal speed) + ~0.8 h evaluation | ~14–17 GB |

Data loading: each worker uses 8 CPU data workers + 4 for the subset (as in round 7).

Disk (all on the HDD, `/DATA1/shrusti/genai_project/results`): the Optuna study keeps no checkpoints (~1 MB + logs). Phase B and the final run: ~0.3 GB each.

## 5. Schedule

| When | What |
|---|---|
| 10-18 ~now | commit, pull, `pip install optuna`, smoke test, **start phase A** |
| 10-19 ~02:00 | phase A done → `--summary`, `--write_configs`, **start phase B** |
| 10-19 ~06:30 | phase B done → decision → **start the final run** (~07:00) |
| 10-20 ~04:00–10:30 | final run done → evaluation; final choice (`PLAN.md` §5): `a29_tuned` replaces a29 only if better at its best noise scale |
| 10-20 | export (fresh noise per call), fvcore + ptflops, CPU test, report; **submit by 23:59** |

Fallbacks: a29 (trained, current final), then a25_full.

## 6. What to send back

- After phase A: `python scripts/tune_optuna.py --results_root $R --summary` output and `$R/optuna/trials_a29_tune.csv`.
- After phase B: `$R/run_round8_phaseB.log`.
- After the final run: `$R/run_round8_final.log`, and for `a29_tuned` (in `$R/a29_tuned/`): `metrics/final_evaluation_summary.json`, `metrics/generative_eval.json`, `logs/training_log.csv`, `visualizations/epoch_0080_comparison.png`.
