# Dataset Notes

## What's used by default and why

| Dataset | Size | Why it was chosen |
|---|---|---|
| **DIV2K_train_HR** | 800 images, 2K resolution | The standard modern benchmark for image restoration training. High resolution means many distinct 256x256 crops per image, and content is broad (people, architecture, nature, animals, text, textures) -- important since the corruption pipeline is randomized on-the-fly, so more distinct source pixels = more effective training variety. |
| **DIV2K_valid_HR** | 100 images | Used as a natural-image validation split -- these images are never seen during training, giving a clean estimate of generalization even within the natural-photo domain. |
| **BSDS500** | ~500 images | Classic restoration/segmentation benchmark with different photographic style and content distribution than DIV2K (more classic photography compositions, different camera characteristics) -- adds scene diversity cheaply (~70MB). |

Together these give roughly **1,300+ diverse natural images**, all
license-appropriate for research/competition use (both are standard,
widely-used research benchmark datasets, satisfying the brief's Section
3.1 requirement).

Both are downloaded via direct, stable institutional URLs (ETH Zürich's
Computer Vision Lab for DIV2K, UC Berkeley's Computer Vision group for
BSDS500) and cached locally after the first download -- see
`src/common/dataset.py::ensure_datasets()`.

## Why this combination, specifically, for THIS competition

The held-out evaluation set (brief Section 4) spans three things at once:
1. **Diverse degradation types and severities** (some undisclosed) →
   addressed by the **randomized on-the-fly degradation pipeline**
   (`src/common/degradations.py`), not by the dataset choice itself. See
   that file's docstring for the reasoning (based on the literature
   review's GenDeg discussion): broaden and randomize corruptions rather
   than fix a small preset list.
2. **Diverse image content** (natural photography) → addressed by
   DIV2K + BSDS500's combined ~1,300 images spanning many scene types.
3. **Diverse *domains*** (natural / low-light / medical) → **only
   partially** addressed by DIV2K/BSDS500 alone, since both are natural
   daylight photography. See the extension section below.

## Extending to low-light and medical domains (recommended, optional)

The brief's Section 4 explicitly states the held-out set includes
low-light imagery and medical imaging (X-ray, MRI). Since DIV2K/BSDS500
alone are natural daylight photos, adding even a modest slice of these
other domains to your training data is likely to measurably help
cross-domain transfer (this is a "domain generalization via broad
training data" strategy, not test-time domain adaptation -- see the
competition-clarification discussion in the project's parent conversation
for why domain *adaptation* isn't applicable here).

**How to add extra domains:**
1. Collect a modest number of license-appropriate images (a few hundred
   is enough to help; you don't need thousands) for each extra domain you
   want to cover. Good starting points:
   - **Low-light**: the LOL (Low-Light) dataset is the standard academic
     benchmark for this; search "LOL dataset low-light image enhancement"
     for current mirror links, since hosting has moved over time. Any
     Creative-Commons low-light photo collection also works.
   - **Medical (X-ray)**: NIH's publicly released Chest X-ray datasets
     (e.g. via Kaggle, search "NIH Chest X-ray dataset") are a common,
     clearly-licensed starting point.
   - **Medical (MRI)**: several open MRI datasets exist on Kaggle /
     academic mirrors (search "brain MRI dataset public domain") -- pick
     one with an explicit open license.
2. Place the images at:
   ```
   <DATA_ROOT>/extra/<domain_name>/*.png   (or .jpg)
   ```
   e.g. `data/extra/lowlight/*.png`, `data/extra/chest_xray/*.png`.
3. Re-run `ensure_datasets()` (or just re-run the training notebook) --
   it automatically detects any folders under `extra/` and includes them
   in the training pool. No code changes needed.

This step is **entirely optional** -- the pipeline trains and evaluates
correctly without it, using DIV2K + BSDS500 alone. Adding extra domains is
a lever to pull if you have time and want to push cross-domain
generalization further, not a requirement to get a working submission.

## A note on patch cropping and content diversity

`DenoisingDataset` (in `src/common/dataset.py`) takes a random 256x256
crop from each source image on every access, with random flips/rotations
applied too. Since DIV2K images are ~2K resolution, this means a single
epoch sees a different crop of each image than the epoch before -- this
is a cheap form of data augmentation that multiplies the effective
dataset size well beyond the raw ~1,300 image count, without needing more
storage or download.
