# Walkthrough: round 6 (a25_full, a27, a28)

Written 2026-10-15. What was just added, how to get it onto the GPU machine (two 24 GB RTX A5000), and the command that runs everything. Plan and reasoning: `PLAN.md`. Model details: `architectures/a25_full.md`, `a27_gan_wave_varctl.md`, `a28_gan_wave_2stage.md`. Round-5 results: `COMPARISON.md` §10.

## 1. What was completed

Three runs, ready to train. Same data (pipeline v2), benchmarks, generator, and wavelet-domain discriminator as a25; all from scratch.

- **a25_full**: a25 unchanged, **200k steps**. The compliant fallback (a25 is the best eligible model so far). 4.98 GFLOPs.
- **a27_gan_wave_varctl**: the round-6 recipe. 4.98 GFLOPs.
- **a28_gan_wave_2stage**: a27 with the adversarial term (and discriminator training) switched on only at 50% of training; the diversity constraint is on from step 0. 4.98 GFLOPs.

The round-6 recipe (a27, a28), which addresses round 5's ~1 dB loss and over-smoothing:
- **16 different images per step** again (round 5: 8), and a **second noise draw for 4 of them** (20 generator outputs).
- **Full composite loss + wavelet L1 on every output** (round 5 applied them to the mean of two samples, which halved each sample's reconstruction signal).
- **Diversity hinge** on the 4 pairs: target 1.5 grey levels, weight 4.0, so it stays active and keeps the model sampling.
- Adversarial weight back to 0.005 (round 5: 0.01).

Submission change: **the exported model now draws fresh noise on every call**: `denoise_image(img)` gives a new sample each time, and `denoise_image(img, seed=k)` gives a reproducible one. This makes the generative behaviour visible to anyone running it. No FLOPs change.

Files added or changed:
- `src/common/gan.py`: pair mode (`stochastic.pair_images`), delayed adversarial phase (`gan.adv_start_frac`), discriminator step factored into `_disc_step`. Round-4/5 configs behave as before.
- `src/common/gennet.py`, `src/common/resdiff.py`: `fresh` noise mode, `set_fresh_noise()`.
- `scripts/export_inference.py`: `seed` argument and fresh noise in the submission's `inference.py`; the package check uses seed 0.
- `src/architectures/a25_full/`, `a27_gan_wave_varctl/`, `a28_gan_wave_2stage/`: `config.yaml` + `model.py`; `src/registry.py`: three entries.
- Docs: `PLAN.md`, `CURRENT_STATUS.md`, `COMPARISON.md` §10, `PROJECT_REVIEW_AND_OPTIONS.md` (Options H–K), `architectures/` (a24–a26 results; a25_full, a27, a28 new; README).

Tested on CPU: 4 training steps of a27, a28, a25_full and a24 (round-5 path unchanged) with all losses (DISTS off, offline). a28's adversarial and discriminator losses are 0 before the switch and active after it. The sample spread is reported in every step. Seeded output is reproducible, fresh mode gives a new output each call, and switching back restores the seeded output. The export template renders and parses. **Not tested on a GPU**: mixed precision, DISTS, memory, speed. The dry run covers these.

## 2. Commit and push (on this machine)

Three commits: round-5 results and analysis, round-6 code, round-6 docs.

```
# 1. round-5 results and analysis
git add results/a24_gan_div results/a25_gan_div_wave results/a26_resdiff_gan results/comparison_round5
git add docs/COMPARISON.md docs/PROJECT_REVIEW_AND_OPTIONS.md docs/architectures/a24_gan_div.md docs/architectures/a25_gan_div_wave.md docs/architectures/a26_resdiff_gan.md
git commit -m "docs(results): add round 5 (a24-a26) results and analysis; new options H-K"

# 2. round-6 code
git add src scripts
git commit -m "feat: add round 6 models a25_full, a27, a28 (full per-sample loss, diversity constraint, delayed adversarial); fresh noise per call in export"

# 3. round-6 plan and docs
git add docs
git commit -m "docs: round 6 plan, status, architecture docs and walkthrough"

git status
git push origin main
```

`git status` at the end should show nothing to commit. If it still shows deleted files under `results/`, they come from the earlier clean-up commit; leave them unless you meant to delete more.

## 3. Run on the GPU machine

```
cd <repo on the GPU machine>
git pull origin main
```

**Dry run first** (about 15 minutes; 30 training steps per model, then the real evaluate, generative-eval and export steps; prints speed and `gpu_mem`):

```
python scripts/run_all.py --data_root <DATA> --archs a25_full a27_gan_wave_varctl a28_gan_wave_2stage --dry_run
```

All three must say `OK`. In a28's dry run, `gen_loss` and `disc_loss` are 0: correct, its adversarial phase starts at 50%.

**Real run, one command** (both GPUs; queue order matters):

```
nohup python scripts/run_all.py --data_root <DATA> --archs a25_full a27_gan_wave_varctl a28_gan_wave_2stage > run_round6.log 2>&1 &
```

GPU 0 takes a25_full (~17 h), GPU 1 takes a27 (~7 h) and then a28 (~6.5 h). For each model: train → evaluate (v1, v2) → generative eval (diversity, noise ×0.5/×1/×1.5) → export. The final comparison table is printed when all three are done (~17.5 h); a27 and a28 have their metrics in `results/<model>/metrics/` as soon as each finishes. If interrupted, run the same command again: finished models are skipped, unfinished ones resume.

While it runs:

```
tail -f run_round6.log
tail -f results/a27_gan_wave_varctl/logs/console.log      # spread255 in the step lines
column -s, -t < results/a27_gan_wave_varctl/logs/training_log.csv | tail -5
nvidia-smi
```

**Checks:**
- **After ~1 h of a27:** `spread255` should sit near 1.5. If it is below 0.5 and falling, the noise is being switched off: tell me (the diversity weight goes up).
- **a27 at epoch 4:** val PSNR should be ~30.3–30.5 (a20 was 30.47 at epoch 4, a25 29.53). If it is still ~29.5, the fix did not work.
- **a28 at epoch 12** (end of phase 1): PSNR near a20's 31.65 at the same point; watch the change after the switch.

## 4. Time and memory (two 24 GB GPUs)

Times from round 5 (measured: a25 5.5 h for 60k incl. evaluation, a24 6.8 h). a27/a28 make 20 outputs per step instead of 16 (~1.25×); a28's first half has no discriminator (faster). Memory was not logged in round 5; the dry run prints the real peak.

| Run | GPU | Steps | Time | Peak memory (of 24 GB) |
|---|---|---|---|---|
| a25_full | 0 | 200k | ~17 h + 0.5 h eval | ~12–14 GB |
| a27_gan_wave_varctl | 1 | 60k | ~6.5–7 h + 0.5 h eval | ~14–17 GB |
| a28_gan_wave_2stage | 1 (after a27) | 60k | ~6 h + 0.5 h eval | ~11–13 GB in phase 1, ~14–17 GB in phase 2 |

- **Started this afternoon:** a27 is done around 23:00–midnight, a28 the next morning (~07:00), a25_full ~08:00–10:00.
- **Disk:** ~0.3 GB per model, ~1 GB total.
- If the dry run shows more than ~22 GB for a27/a28, add `--train_args "--batch_size 12"` (12 images + 4 extra draws) and tell me.

## 5. What to send back

`run_round6.log` (or, if you want to decide earlier, a27's and a28's `metrics/` as soon as each finishes). Per model:

- `results/<model>/metrics/final_evaluation_summary.json`
- `results/<model>/metrics/generative_eval.json`
- `results/<model>/metrics/final_report.json`
- `results/<model>/logs/training_log.csv`
- `results/<model>/visualizations/epoch_00XX_comparison.png` (last epoch)
- `results/comparison/` (CSV files)

First numbers to look at: a27/a28 PSNR against **31.33** (a25 + 0.4, the rule for a 200k run), and diversity ≥ 0.5.

## 6. Schedule

| Day | What |
|---|---|
| 2 (10-15) | commit, pull, dry run, start round 6 |
| 3 (10-16) | morning: a28 and a25_full done → send results; decision by `PLAN.md` §6. If a27/a28 qualifies: its 200k config (small step from me), started by noon (~21 h) |
| 4 (10-17) | 200k run done (~10:00) → evaluation; final model chosen; tuning (decided then) on the free GPU |
| 5 (10-18) | export (fresh noise per call), FLOPs with fvcore + ptflops, CPU test, diversity proof |
| 6 (10-19) | report |

If neither a27 nor a28 qualifies, a25_full is the final model and both GPUs are free from Day 3 for tuning or the v3 fine-tune.
