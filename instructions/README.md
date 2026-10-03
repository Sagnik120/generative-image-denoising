# instructions/ — How to Run

Practical, step-by-step "how do I actually run this" guides. For the
*reasoning* behind each architecture's design, see `docs/` instead.

Architectures are numbered in the order they were added (`a01` … `a11`; the
highest number is the newest). `results/INDEX.md` lists them all with their
headline results. Round 1 (`a01`–`a04`) used the original DIV2K-only
pipeline and is kept for reference; round 2 (`a05` onward) uses the
multi-dataset pipeline and the fixed benchmark.

## Server workflow (round 2)

Run everything from the project root, inside the virtual environment.

### 1. One-time setup
```bash
git pull
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare the data (once, ~30 GB download)
```bash
python scripts/prepare_data.py --check_urls            # 10 s: is every dataset reachable?
nohup python scripts/prepare_data.py --data_root /path/to/data > prepare_data.log 2>&1 &
tail -f prepare_data.log
```
It ends with `DONE. Data is ready for training.` If a download breaks, run
the same command again: finished datasets are skipped and partial downloads
resume. It exits with an error listing any dataset that is still incomplete,
and training refuses to start until none is.

### 3. CPU smoke test (under a minute)
```bash
python scripts/verify_all.py
```

### 4. Rehearse on the GPUs (a few minutes per architecture)
```bash
python scripts/run_all.py --data_root /path/to/data --dry_run
```
Runs 30 real training steps, a short validation, a checkpoint, the real
evaluate step and the real export step for every architecture, and prints
each one's speed (it/s) and peak GPU memory. Nothing is written to the real
results. Do not start the long run until every line says `OK`.

### 5. Train everything with one command
```bash
nohup python scripts/run_all.py --data_root /path/to/data > run_all.log 2>&1 &
```
- Uses every idle GPU of at most 30 GB (so never the 48 GB card), one
  architecture per GPU at a time, picked by GPU UUID.
- Each architecture runs train → evaluate → export in its own process; if
  one fails, that GPU moves on to the next.
- When the command exits it prints `ALL DONE` and every GPU it used is free.
- Interrupted? Run the same command again: finished architectures are
  skipped, unfinished ones resume from their last checkpoint.

To run a subset, or choose GPUs by their `nvidia-smi` index:
```bash
nohup python scripts/run_all.py --data_root /path/to/data --archs a07_naf_attn_hybrid a08_degradation_aware --gpus 1 2 > run_all.log 2>&1 &
```

### 6. Watch progress
```bash
tail -f run_all.log                                    # which architecture is on which GPU
tail -f results/a07_naf_attn_hybrid/logs/console.log   # one architecture's training log
gpustat
```

### 7. Results
- `results/INDEX.md` — every architecture, numbered, with its four metrics
- `results/comparison/comparison_table.csv`, `comparison_ranking.csv` — the four axes and per-axis ranks
- `results/comparison/comparison_by_group.csv`, `comparison_by_case.csv` — results per degradation group, domain and individual combination
- `results/<arch>_results.zip` — one architecture's logs, metrics and figures, for copying off the server

### Running a single architecture by hand
```bash
CUDA_VISIBLE_DEVICES=1 nohup python scripts/train.py --arch a07_naf_attn_hybrid --data_root /path/to/data --resume auto > output_a07.log 2>&1 &
python scripts/evaluate.py --arch a07_naf_attn_hybrid --data_root /path/to/data
python scripts/export_inference.py --arch a07_naf_attn_hybrid
python scripts/compare_architectures.py
```
`train.py` prints the GPU it landed on and refuses to run on one larger than
30 GB unless `--allow_big_gpu` is passed.

### Rules while runs are active
- Do not `git pull` or edit files under `src/` — running processes import
  the code as it was at start, but a resumed or later-queued run would pick
  up the change mid-comparison.
- Do not run `prepare_data.py` with `--rebuild_benchmark`: every
  architecture must be scored on the same benchmark.
- Two processes cannot train the same architecture into the same results
  folder; the second one exits immediately.

## Per-architecture guides
- Round 2: `a05_nafnet_v2/`, `a06_restormer_v2/`, `a07_naf_attn_hybrid/`,
  `a08_degradation_aware/`, `a09_wavelet_naf/`, `a10_resflow_fewstep/`,
  `a11_freq_spatial/`
- Round 1: `a01_nafnet_unet/`, `a02_restormer_lite/`, `a03_pix2pix_gan/`,
  `a04_tiny_ddpm_sr3/`

Each covers that architecture's own commands and where its results land.
