# Dataset, Degradation and Benchmark Notes

## Why round 1's data was not enough
Round 1 intended DIV2K + BSDS500, but the BSDS500 URL had gone dead and the
code fell back to DIV2K alone with only a warning. All four models were
therefore trained on 800 daylight photographs, cropped at native 2K zoom,
while the test set is whole images shown at 256x256 from three domains.
Round 2 fixes the content, the scale and the silent fallback.

## Sources (all fetched by `scripts/prepare_data.py`, ~30 GB)

| Domain | Source | Size | Used for |
|---|---|---|---|
| Natural | DIV2K train (800 images, 2K) | 3.5 GB | training |
| Natural | Flickr2K (2,650 images, 2K) | 11.6 GB | training |
| Natural | BSDS500 (300 train+val / 200 test) | 0.2 GB | training / benchmark |
| Natural | COCO test2017 (~40k everyday scenes) | 6.6 GB | training |
| Natural | DIV2K valid (100), COCO val2017 (300 used) | 1.3 GB | benchmark only |
| Low-light | LOL (485 train / 15 eval pairs) | 0.35 GB | training / benchmark |
| X-ray | NIH ChestX-ray, first archive (~5,000 images) | 2.0 GB | training (last 80 held out) |
| MRI | IXI T2 brain volumes (~580 subjects) | 3.9 GB | training (last 30 subjects held out) |
| CT | MedMNIST OrganC, 224 px | 0.8 GB | training / benchmark |
| Ultrasound | MedMNIST Breast, 224 px | 0.03 GB | training / benchmark |

Every source has its mirrors listed in `src/common/dataset.py::SOURCES`;
downloads resume, each mirror is tried in turn, and `manifest.json` records
the item count of every source. Training refuses to start while any training
view is missing. Check each dataset's licence terms against the brief's
Section 3.1 before submitting; all are public research datasets.

## How they are combined
Sources are **sampled by weight**, not concatenated, so 40k COCO images
cannot drown out 600 ultrasound images. Weights are in
`src/common/dataset.py::TRAIN_VIEWS`; roughly 70% natural, 20% medical, and
about 13% low-light (LOL plus natural photos darkened synthetically).

Each training sample is produced as follows:
1. Pick a view by weight, then an image from it.
2. Shrink it so its short side is close to 256, then take a random 256 crop.
   The test set is whole images at 256x256, so most samples are whole-scene
   views; only the `natural_tiles` view keeps native 2K detail.
3. Random flips and rotations; 4% of natural samples are converted to
   greyscale, 10% are darkened into low-light images.
4. Apply a random degradation chain (below) and snap the result to uint8.

2K photographs are converted once into a 512 px whole view plus native
512 px tiles, MRI volumes into slices, so training never decodes a 2K image
to take one crop.

## Degradations (`src/common/degradations.py`)
- **Listed families** (the brief's nine): Gaussian, Poisson, salt-and-pepper
  and speckle noise; Gaussian, motion and defocus blur; JPEG; chromatic
  aberration.
- **Extra families** also trained on: spatially correlated noise,
  random-valued impulses, Rician noise, stripes/banding, resize blur, WebP,
  posterisation, and scratches / dead pixels / small blocks.
- **Unseen families**, never trained on, used only by the benchmark:
  uniform noise, nearest-neighbour pixelation, periodic interference.

Each sample gets a chain of 0, 1, 2, 3 or 4 families (about 3 / 34 / 39 /
19 / 5 %), each with a continuous severity. Chains are applied in physical
order (optics, blur, noise, compression) with 20% shuffled. Longer chains
draw gentler individual severities so a four-family mix is still
restorable. Three quarters of the draws come from the listed families.
Brightness/contrast change is deliberately not a degradation: the model
cannot know the original exposure, so "undoing" it only costs PSNR.

## The fixed benchmark
`prepare_data.py` builds it once under `<data_root>/benchmark/v1/`: 33 cases
x 12 held-out images (6 natural, 3 low-light, 3 medical) = 396 pairs, each
with its corruption generated once and saved to disk.

| Group | Cases |
|---|---|
| single | each of the nine listed families |
| extra | correlated noise, resize blur, small artifacts |
| pair | twelve combinations (blur+noise, noise+JPEG, blur+JPEG, ...) |
| triple | five combinations (blur+noise+JPEG, ...) |
| unseen | three families never trained on, and one mix of two of them |

Every architecture is validated on these same 396 pairs after every epoch,
and `scripts/evaluate.py` reports PSNR / SSIM / DISTS overall and per group,
domain and case. Round 1 re-randomised its validation images every epoch,
which moved PSNR by about 1 dB between epochs and made its comparison
unreliable.
