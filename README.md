# Generative Image Denoising — Mini Competition Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sagnik120/generative-image-denoising/blob/main/notebooks/train_colab.ipynb)

A scalable, multi-architecture training framework for the "Small
Generative Model for Image Denoising" mini competition. Built to let you
train, evaluate, and compare **multiple generative architectures** against
the same datasets, degradation pipeline, fixed benchmark and metrics — then
pick the strongest submission across all four scored axes (FLOPs, PSNR,
SSIM, DISTS).

See `Generative_Denoising_Literature_Review.md` for the research this
project's architecture and dataset choices are based on.

## Architectures

Numbered in the order they were added; the highest number is the newest.
`results/INDEX.md` (written by `scripts/compare_architectures.py`) lists
every one with its headline results.

**Round 2** — multi-dataset pipeline, about 5 GFLOPs each, scored on the
fixed benchmark. These are the ones to train and compare.

| # | Architecture | Idea |
|---|---|---|
| 05 | `a05_nafnet_v2` | Pure-convolution reference (NAF blocks) |
| 06 | `a06_restormer_v2` | Pure-transformer reference (channel attention) |
| 07 | `a07_naf_attn_hybrid` | Convolution at high resolution, global attention at the 16 px bottleneck |
| 08 | `a08_degradation_aware` | a07 conditioned on a learned degradation embedding |
| 09 | `a09_wavelet_naf` | NAF U-Net in the Haar wavelet domain |
| 10 | `a10_resflow_fewstep` | Generative residual flow, two passes |
| 11 | `a11_freq_spatial` | Spatial and Fourier-domain paths in every block |

**Round 1** — original DIV2K-only pipeline, 11–37 GFLOPs, scored on a
different validation set. Kept for reference; not comparable with round 2.

| # | Architecture | PSNR | SSIM | DISTS | GFLOPs |
|---|---|---|---|---|---|
| 01 | `a01_nafnet_unet` | 27.32 | 0.752 | 0.228 | 12.8 |
| 02 | `a02_restormer_lite` | 27.44 | 0.758 | 0.235 | 11.3 |
| 03 | `a03_pix2pix_gan` | 26.54 | 0.720 | 0.246 | 12.8 |
| 04 | `a04_tiny_ddpm_sr3` | 10.01 | 0.086 | 0.522 | 37.3 (8 steps) |

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
│   │   ├── layers.py                   NAFBlock, attention, Fourier and wavelet blocks
│   │   ├── unet.py                     the U-Net shell shared by round-2 architectures
│   │   ├── bundle.py                   round-2 training logic (AMP, EMA, schedule, loss)
│   │   ├── degradations.py             all corruption families + RandomDegradation + benchmark cases
│   │   ├── dataset.py                  download/caching, mixed-domain dataset, fixed benchmark
│   │   ├── metrics.py                  PSNR / SSIM / DISTS / FLOPs + benchmark breakdown
│   │   ├── losses.py                   Charbonnier / FFT / SSIM / DISTS / GAN losses
│   │   ├── trainer.py                  architecture-agnostic training loop
│   │   ├── visualize.py                loss curves, metric curves, comparison grids
│   │   └── utils.py                    seeding, checkpoints, CSV logging, GPU guard
│   │
│   ├── registry.py                  <- maps architecture name -> build() function
│   │
│   └── architectures/
│       └── a01_... to a11_...       each folder: model.py + config.yaml
│
├── scripts/
│   ├── prepare_data.py              <- download + cache all datasets, build the benchmark
│   ├── run_all.py                   <- train + evaluate + export a queue of architectures
│   ├── train.py                     <- train ONE architecture
│   ├── evaluate.py                  <- benchmark evaluation + zip results/<arch>/
│   ├── export_inference.py          <- package submission-ready weights + inference.py
│   ├── compare_architectures.py     <- comparison tables + results/INDEX.md
│   ├── verify_all.py                <- fast CPU smoke test of every architecture
│   └── verify_trainer_e2e.py        <- end-to-end rehearsal on CPU
│
├── notebooks/
│   └── train_colab.ipynb            <- Colab alternative to the server workflow
│
├── results/                          <- one subfolder per architecture
│   ├── INDEX.md                     numbered list of every architecture and its results
│   ├── a05_nafnet_v2/
│   │   ├── checkpoints/       (best_model.pt, last_model.pt)
│   │   ├── loss_curves/       (loss_curve.png)
│   │   ├── metrics/           (final_evaluation_summary.json, benchmark_breakdown.csv, ...)
│   │   ├── visualizations/    (epoch_XXXX_comparison.png, error heatmaps)
│   │   ├── logs/              (training_log.csv, console.log, run_config.json)
│   │   └── submission/        (created by export_inference.py)
│   ├── ...                    (same structure for every architecture)
│   └── comparison/            (created by compare_architectures.py)
│
├── instructions/                     <- practical "how to run" guides
│   ├── README.md                    the full server workflow
│   └── <arch_name>/README.md        (one per architecture)
│
├── docs/                             <- design rationale / research notes
│   ├── README.md
│   ├── dataset_notes.md             datasets, mixing, degradations, benchmark
│   └── <arch_name>/notes.md         (one per architecture)
│
└── data/                             <- default dataset location (use --data_root for another disk)
```

## Quick start

Full details are in `instructions/README.md`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/verify_all.py                                        # CPU smoke test
python scripts/prepare_data.py --data_root /path/to/data            # once, ~30 GB
python scripts/run_all.py --data_root /path/to/data --dry_run       # rehearse on the GPUs
nohup python scripts/run_all.py --data_root /path/to/data > run_all.log 2>&1 &
```

`run_all.py` trains, evaluates and exports every round-2 architecture, one
per free GPU at a time, and finishes by writing the comparison tables under
`results/comparison/` and `results/INDEX.md`.

## How the four scored metrics are targeted

| Metric | How |
|---|---|
| PSNR | Charbonnier pixel loss plus a Fourier-spectrum loss |
| SSIM | An SSIM loss using the same 7x7 window as `skimage` |
| DISTS | The DISTS metric itself as a loss, in the second half of training |
| FLOPs | Every round-2 architecture is sized to about 5 GFLOPs with fvcore |

Validation and evaluation compute all three image metrics on uint8 images,
as the brief specifies.

## How generalization to unseen degradations and combinations is handled

See `src/common/degradations.py` and `docs/dataset_notes.md`. Each training
image gets a random chain of 0–4 corruptions with continuous severities,
drawn from the brief's nine families plus eight further ones, applied in
physical order. Three more families are never trained on and appear only in
the fixed benchmark, which reports results per individual degradation, per
combination, and for those unseen families.

## How cross-domain generalization (natural/low-light/medical) is handled

Training samples are drawn by weight from natural (DIV2K, Flickr2K,
BSDS500, COCO), low-light (LOL, plus synthetic darkening) and medical
(X-ray, MRI, CT, ultrasound) sources; the benchmark reports each domain
separately. See `docs/dataset_notes.md`.

## Adding a new architecture later

1. Create `src/architectures/aNN_<name>/model.py` (next free number)
   exposing `build_model(model_cfg)` and `build(cfg, device)`; the simplest
   route is a network built on `src/common/unet.py` wrapped in
   `src/common/bundle.py::RestorationBundle` (see `a05_nafnet_v2/model.py`).
2. Create `src/architectures/aNN_<name>/config.yaml`.
3. Add one line to `ARCHITECTURES` in `src/registry.py`.
4. Add `instructions/aNN_<name>/README.md` and `docs/aNN_<name>/notes.md`.

Nothing else changes — the Trainer, dataset pipeline, metrics, evaluation
script and `run_all.py` are all architecture-agnostic.
