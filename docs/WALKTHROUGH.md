# Walkthrough: round 4 (generative models)

Written 2026-10-14. What was just added, how to get it onto the GPU machine, and the one command that runs everything. Plan and reasoning: `PLAN.md`. Model details: `architectures/a20_gan.md` to `a23_resdiff_2step.md`.

## 1. What was completed

Four strictly generative models, ready to train. All use the existing data, degradation pipeline v2, benchmarks and composite loss, and train from scratch for 60k steps.

- **a20_gan**: conditional GAN. NAF U-Net generator with noise injection, U-Net discriminator on RGB. 4.98 GFLOPs.
- **a21_gan_wavelet**: same GAN, but the discriminator sees high-frequency wavelet bands, plus a wavelet fidelity loss. 4.98 GFLOPs.
- **a22_resdiff**: residual diffusion (starts from the corrupted image + noise), 1 sampling step. 5.02 GFLOPs.
- **a23_resdiff_2step**: residual diffusion at width 16, 2 sampling steps. 4.61 GFLOPs.

Files added or changed:

- `src/common/gennet.py`: generator and seeded noise injection.
- `src/common/gan.py`: discriminator, hinge + R1 losses, wavelet loss, GAN training step.
- `src/common/resdiff.py`: diffusion network, training, few-step sampler.
- `src/architectures/a20_gan/`, `a21_gan_wavelet/`, `a22_resdiff/`, `a23_resdiff_2step/`: a `config.yaml` and a short `model.py` each.
- `scripts/eval_generative.py` (new): sample diversity over 8 noise seeds; for diffusion, scores at 1, 2 and 4 steps.
- `scripts/run_all.py`: now runs `eval_generative.py` automatically for these models and prints its table at the end. Also has `--jobs_per_gpu` (default 1; leave it).
- `src/registry.py`: the four new entries.

Tested so far: on CPU only (a few training steps per model, checkpoint save/load, export-style rebuild, reproducible output, FLOPs). **Not yet tested on a GPU**: mixed precision, the data pipeline and the export step. The dry run in step 3 covers these, so do not skip it.

## 2. Commit and push (on this machine)

Only code and docs are added here. The `results/` folder has other uncommitted files (round-3 outputs); they are left out on purpose, add them separately if you want them in the repo.

```
git add src scripts docs
git status
git commit -m "feat: add round 4 generative models a20-a23 (GAN, wavelet GAN, residual diffusion)"
git push origin main
```

## 3. Run on the GPU machine

```
cd <repo on the GPU machine>
git pull origin main
```

**Dry run first** (about 10–15 minutes; 30 training steps per model, then the real evaluate, generative-eval and export steps; prints speed and peak memory per model):

```
python scripts/run_all.py --data_root <DATA> --archs a21_gan_wavelet a22_resdiff a20_gan a23_resdiff_2step --dry_run
```

Check that all four say `OK`. If one says `FAILED`, send me `results/_dry_run/<model>/logs/console.log`.

**Real run, one command:**

```
nohup python scripts/run_all.py --data_root <DATA> --archs a21_gan_wavelet a22_resdiff a20_gan a23_resdiff_2step > run_round4.log 2>&1 &
```

This uses both GPUs, one model per GPU. For each model it runs: train → evaluate (benchmarks v1 and v2) → generative evaluation (diversity, step counts) → export. When a GPU finishes a model it takes the next one. At the end it prints the comparison table and the generative-evaluation table into `run_round4.log`. Nothing else needs to be run by hand.

Useful while it runs:

```
tail -f run_round4.log                              # start/end of each model
tail -f results/a21_gan_wavelet/logs/console.log    # progress of one model
```

If it is interrupted, run the same `nohup` command again: finished models are skipped and unfinished ones resume from their last checkpoint.

## 4. What it will cost

One model per GPU. Only the 10.5 GB figure is measured (the regression model of the same size, in an earlier dry run); the rest are estimates that the dry run will replace with real numbers.

GPU memory (peak, of 24 GB)
- a20, a21 (GAN): about 13–15 GB
- a22 (diffusion, width 24): about 11–13 GB
- a23 (diffusion, width 16): about 8–9 GB

Time
- a20, a21: about 6.5–7.5 h each
- a22: about 5 h
- a23: about 4 h
- Evaluation and export add about 10–15 minutes per model.
- **Whole queue on two GPUs: about 12 h.** GPU 0: a21 then a20 or a23; GPU 1: a22 then the other.

Disk
- About 0.3 GB per model (two checkpoints, submission package, images, logs): about 1.2 GB for all four. No new data is downloaded.

## 5. What to send back

`run_round4.log` is enough for a first look (it contains both tables). For the full analysis, per model:

- `results/<model>/metrics/final_evaluation_summary.json`
- `results/<model>/metrics/generative_eval.json`
- `results/<model>/metrics/final_report.json`
- `results/<model>/logs/training_log.csv`
- `results/<model>/visualizations/epoch_0024_comparison.png`
- `results/comparison/` (the four CSV files)

The number to look at first is **diversity** in the generative table. If it is near zero for a model, that model ignores its noise and is not eligible (plan section 7).

## 6. Schedule for the remaining 5 days

One day is kept free for the report and final checks, so four days of work.

- **Day 1:** commit, pull, dry run, start the screening run (about 12 h, can run overnight).
- **Day 2 morning:** I analyse the results, pick the winner by the rules in the plan, and prepare the final model's config (the same model, 200k steps). This needs one small code step from me (a new config; a diffusion + GAN model only if both families show problems).
- **Day 2 to Day 3:** final long run. About 17–25 h depending on the winner (a GAN is the slow case). The second GPU is free during this time and can run one tuning variant of the winner at 60k steps (for example a different adversarial weight or noise scale).
- **Day 4:** evaluate the final model, clean-input check, export, FLOPs with both counters, CPU inference test. If time allows, one short fine-tune with a wider corruption list.
- **Day 5:** buffer for anything that failed, then freeze the submission.
- **Day 6:** report and final packaging.

There is no room in this schedule for a second full screening round, so if the dry run shows a problem it is worth fixing it the same day.
