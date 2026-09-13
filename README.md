# Generative Image Denoising — Mini Competition Project

A scalable, multi-architecture training framework for the "Small
Generative Model for Image Denoising" mini competition. Built to let you
train, evaluate, and compare **multiple generative architectures** against
the same dataset, degradation pipeline, and metrics — then pick the
strongest submission across all four scored axes (FLOPs, PSNR, SSIM, DISTS).

See `Generative_Denoising_Literature_Review.md` for the research this
project's architecture and dataset choices are based on.

## Project structure

```
.
├── README.md                        <- this file
├── requirements.txt
├── Mini_Competition.pdf              <- the original competition brief
├── Generative_Denoising_Literature_Review.md
│
├── src/
│   ├── common/                      <- shared, reusable code (used by every architecture)
│   │   ├── layers.py                   NAFBlock, LayerNorm2d, FiLM, etc.
│   │   ├── degradations.py             all corruption functions + RandomDegradation
│   │   ├── dataset.py                  dataset download/caching + paired Dataset
│   │   ├── metrics.py                  PSNR / SSIM / DISTS / FLOPs
│   │   ├── losses.py                   Charbonnier / L1 / L2 / GAN losses
│   │   ├── trainer.py                  architecture-agnostic training loop
│   │   ├── visualize.py                loss curves, metric curves, comparison grids
│   │   └── utils.py                    seeding, checkpoints, CSV logging
│   │
│   ├── registry.py                  <- maps architecture name -> build() function
│   │
│   └── architectures/
│       ├── nafnet_unet/             <- PRIMARY RECOMMENDATION (lowest FLOPs)
│       ├── restormer_lite/          <- efficient Transformer (higher quality ceiling)
│       ├── pix2pix_gan/             <- conditional GAN (likely best perceptual/DISTS)
│       └── tiny_ddpm_sr3/           <- few-step conditional diffusion (most experimental)
│           each folder: model.py + config.yaml
│
├── scripts/
│   ├── train.py                     <- train ONE architecture at a time
│   ├── evaluate.py                  <- full evaluation + zip results/<arch>/
│   ├── export_inference.py          <- package submission-ready weights + inference.py
│   ├── compare_architectures.py     <- side-by-side comparison across all trained archs
│   ├── verify_all.py                <- fast CPU smoke test of every architecture
│   └── verify_trainer_e2e.py        <- end-to-end Trainer integration test
│
├── notebooks/
│   └── train_colab.ipynb            <- upload this to Colab; trains on GPU, saves to Drive
│
├── results/                          <- one subfolder per architecture (see below)
│   ├── nafnet_unet/
│   │   ├── checkpoints/       (best_model.pt, last_model.pt)
│   │   ├── loss_curves/       (loss_curve.png, gan_loss_curve.png if applicable)
│   │   ├── metrics/           (metrics_dashboard.png, final_report.json, ...)
│   │   ├── visualizations/    (epoch_XXXX_comparison.png, error heatmaps)
│   │   ├── logs/              (training_log.csv)
│   │   └── submission/        (created by export_inference.py)
│   ├── restormer_lite/        (same structure)
│   ├── pix2pix_gan/           (same structure)
│   ├── tiny_ddpm_sr3/         (same structure)
│   └── comparison/            (created by compare_architectures.py)
│
├── instructions/                     <- practical "how to run" guides
│   ├── README.md
│   └── <arch_name>/README.md        (one per architecture)
│
├── docs/                             <- design rationale / research notes
│   ├── README.md
│   ├── dataset_notes.md
│   └── <arch_name>/notes.md         (one per architecture)
│
└── data/                             <- local dataset cache (empty until first run;
                                          use a Google Drive path instead when using Colab)
```

## Quick start

### 1. Local sanity check (no GPU needed, a few seconds)
```bash
pip install -r requirements.txt
python scripts/verify_all.py
```
This confirms every architecture builds and trains correctly on tiny
dummy data before you commit any real GPU time.

### 2. Train on Colab (recommended — free GPU)
1. Zip this entire project folder.
2. Upload the zip to your Google Drive at
   `MyDrive/denoising_competition/project.zip`.
3. Open `notebooks/train_colab.ipynb` in Google Colab (`Runtime -> Change
   runtime type -> GPU`).
4. Run the cells top to bottom. Set `ARCH` in the Configuration cell to
   whichever architecture you want to train this run.
5. The dataset (DIV2K + BSDS500, ~3.7GB) downloads **once** into your
   Drive and is cached — restarting the Colab runtime never re-downloads
   it.
6. At the end, results are zipped in the same folder structure as
   `results/<arch>/` locally — download and unzip directly into your
   local project's `results/` folder.
7. Repeat steps 4–6 with a different `ARCH` to train and compare all four
   architectures.

### 3. Train locally (if you have a GPU)
```bash
python scripts/train.py --arch nafnet_unet --data_root ./data
python scripts/evaluate.py --arch nafnet_unet --data_root ./data
python scripts/export_inference.py --arch nafnet_unet
```

### 4. Compare all trained architectures
```bash
python scripts/compare_architectures.py
```
Produces `results/comparison/comparison_table.csv`,
`comparison_ranking.csv` (mirroring the competition's own independent
per-axis ranking scheme), and a bar-chart dashboard image.

## The four architectures at a glance

| Architecture | Family | Expected strength | Expected FLOPs |
|---|---|---|---|
| `nafnet_unet` | CNN (activation-free) | Best FLOPs efficiency, stable training | Lowest |
| `restormer_lite` | Efficient Transformer | Higher quality ceiling (structure/perceptual) | Medium-high |
| `pix2pix_gan` | Conditional GAN | Best perceptual realism (DISTS) | Same as nafnet_unet at inference (discriminator is training-only) |
| `tiny_ddpm_sr3` | Few-step diffusion | Best theoretical zero-shot generalization | Highest (T forward passes) |

Train all four, evaluate each with `scripts/evaluate.py`, then run
`scripts/compare_architectures.py` to see which wins on your own held-out
validation split before deciding what to submit.

## How generalization to unseen degradations is handled

See `src/common/degradations.py` — a `RandomDegradation` class applies 1–3
randomly chosen, randomly severed corruptions per training image, from a
pool that's deliberately **broader** than the competition brief's
disclosed list (includes a few extra corruption families too), so the
model never overfits to a small fixed set of corruption recipes. This
directly implements the literature review's GenDeg-inspired
recommendation.

## How cross-domain generalization (natural/low-light/medical) is handled

The default dataset (DIV2K + BSDS500) covers natural photography broadly.
See `docs/dataset_notes.md` for how to optionally add low-light and
medical-imaging slices to the training pool — the pipeline auto-detects
any images placed under `data/extra/<domain_name>/`.

## Adding a new (5th, 6th, ...) architecture later

1. Create `src/architectures/<new_name>/model.py` exposing a
   `build(cfg, device)` function returning a `ModelBundle` (see the
   interface documented at the top of `src/common/trainer.py`).
2. Create `src/architectures/<new_name>/config.yaml`.
3. Add one line to `ARCHITECTURES` in `src/registry.py`.
4. (Optional) add `instructions/<new_name>/README.md` and
   `docs/<new_name>/notes.md`.

Nothing else changes — the Trainer, dataset pipeline, metrics, evaluation
script, and Colab notebook are all architecture-agnostic.
