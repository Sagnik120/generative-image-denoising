"""
Synthetic degradation pipeline.

The brief (Section 3.2 / 4) asks for a model that removes many KINDS of
corruption and many COMBINATIONS of them, including kinds that are not
disclosed. This file is where that is decided, so it is organised around
three ideas:

  1. Every degradation takes a severity `s` in [0, 1] and an explicit numpy
     Generator. Severity makes "mild / medium / severe" a controllable axis
     (used by the fixed benchmark); the explicit Generator makes every
     DataLoader worker draw different corruptions (numpy's global RNG is
     copied into each forked worker, which silently repeats the same noise).

  2. Three family lists:
       LISTED_FAMILIES  - the nine named in the brief.
       EXTRA_FAMILIES   - further corruptions we also train on, so the model
                          sees more than the disclosed list.
       UNSEEN_FAMILIES  - NEVER used in training; only the benchmark applies
                          them, to measure generalisation to unknown types.

  3. `RandomDegradation` builds a chain of 0-4 families per image, applied
     in the order a camera would produce them (optics -> blur -> noise ->
     compression), with a fraction of chains shuffled so the model does not
     depend on one order. It also returns a label vector saying which
     families were applied and how severely.

All functions take and return float32 HWC arrays in [0, 1].
"""
import io

import numpy as np
import cv2
from PIL import Image


def _clip(img):
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _lerp(lo, hi, s):
    return lo + (hi - lo) * float(s)


def _to_uint8(img):
    return (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def quantize_uint8(img):
    """Snap to the uint8 grid: the model's real input is always a uint8 image."""
    return _to_uint8(img).astype(np.float32) / 255.0


# ------------------------- noise family ------------------------- #

def add_gaussian_noise(img, s, rng):
    sigma = _lerp(2, 55, s) / 255.0
    shape = img.shape[:2] + (1,) if rng.random() < 0.25 else img.shape   # 25%: same noise on all channels
    return _clip(img + rng.normal(0, sigma, shape).astype(np.float32))


def add_poisson_noise(img, s, rng):
    peak = 10 ** _lerp(3.0, 0.7, s)          # photons at white: 1000 (mild) -> 5 (severe)
    return _clip(rng.poisson(img * peak) / peak)


def add_salt_and_pepper(img, s, rng):
    amount = _lerp(0.002, 0.20, s)
    salt_frac = rng.uniform(0.3, 0.7)
    shape = img.shape if rng.random() < 0.5 else img.shape[:2] + (1,)    # per-channel or whole-pixel
    u = np.broadcast_to(rng.random(shape), img.shape)
    out = img.copy()
    out[u < amount * salt_frac] = 1.0
    out[(u >= amount * salt_frac) & (u < amount)] = 0.0
    return out


def add_speckle_noise(img, s, rng):
    sigma = _lerp(0.03, 0.45, s)
    return _clip(img + img * rng.normal(0, sigma, img.shape).astype(np.float32))


# ------------------------- blur family ------------------------- #

def _filter(img, kernel):
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)


def apply_gaussian_blur(img, s, rng):
    sigma = _lerp(0.4, 4.0, s)
    if rng.random() < 0.7:
        return _clip(cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT))
    # Anisotropic, rotated Gaussian.
    sx, sy = sigma, sigma * rng.uniform(0.3, 0.8)
    theta = rng.uniform(0, np.pi)
    r = int(np.ceil(3 * sigma))
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    kernel = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    return _clip(_filter(img, (kernel / kernel.sum()).astype(np.float32)))


def _line_kernel(length, angle_deg):
    ksize = int(np.ceil(length)) | 1
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    return kernel / max(kernel.sum(), 1e-8)


def _trajectory_kernel(length, rng):
    """Camera-shake style kernel: a random curved path instead of a straight line."""
    n = max(int(length * 4), 8)
    angles = rng.uniform(0, 2 * np.pi) + np.cumsum(rng.normal(0, 0.18, n))
    step = length / n
    xs = np.cumsum(step * np.cos(angles))
    ys = np.cumsum(step * np.sin(angles))
    xs -= (xs.max() + xs.min()) / 2
    ys -= (ys.max() + ys.min()) / 2
    r = int(np.ceil(max(np.abs(xs).max(), np.abs(ys).max()))) + 1
    kernel = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.float32)
    np.add.at(kernel, (np.round(ys).astype(int) + r, np.round(xs).astype(int) + r), 1.0)
    return kernel / kernel.sum()


def apply_motion_blur(img, s, rng):
    length = _lerp(3, 25, s)
    if rng.random() < 0.7:
        kernel = _line_kernel(length, rng.uniform(0, 180))
    else:
        kernel = _trajectory_kernel(length, rng)
    return _clip(_filter(img, kernel))


def apply_defocus_blur(img, s, rng):
    radius = _lerp(0.8, 8.0, s)
    r = int(np.ceil(radius)) + 1
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    kernel = np.clip(radius + 0.5 - np.sqrt(x ** 2 + y ** 2), 0.0, 1.0)   # soft-edged disk
    return _clip(_filter(img, (kernel / kernel.sum()).astype(np.float32)))


# ------------------------- compression / colour ------------------------- #

def _pil_roundtrip(img, fmt, **save_kwargs):
    buf = io.BytesIO()
    Image.fromarray(_to_uint8(img)).save(buf, format=fmt, **save_kwargs)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0


def apply_jpeg_compression(img, s, rng):
    quality = int(round(_lerp(95, 8, s)))
    return _pil_roundtrip(img, "JPEG", quality=quality, subsampling=int(rng.choice([0, 1, 2])))


def apply_chromatic_aberration(img, s, rng):
    """Shifts and slightly rescales the red and blue planes against green."""
    h, w = img.shape[:2]
    max_shift = _lerp(0.5, 6.0, s)
    out = img.copy()
    radial = rng.random() < 0.5
    for ch, sign in ((0, 1.0), (2, -1.0)):
        ang = rng.uniform(0, 2 * np.pi)
        mag = rng.uniform(0.4, 1.0) * max_shift
        scale = 1.0 + sign * mag / (w / 2) if radial else 1.0
        tx = 0.0 if radial else mag * np.cos(ang)
        ty = 0.0 if radial else mag * np.sin(ang)
        M = np.float32([[scale, 0, tx + (1 - scale) * w / 2],
                        [0, scale, ty + (1 - scale) * h / 2]])
        out[..., ch] = cv2.warpAffine(img[..., ch], M, (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT)
    return _clip(out)


# ------------------------- extra families (trained on) ------------------------- #

def add_correlated_noise(img, s, rng):
    """Spatially correlated (blotchy) noise, as left by demosaicing / denoisers."""
    sigma = _lerp(2, 45, s) / 255.0
    noise = rng.normal(0, 1, img.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), rng.uniform(0.6, 1.6))
    return _clip(img + sigma * noise / max(noise.std(), 1e-6))


def add_random_impulse(img, s, rng):
    """Random-valued impulse noise: corrupted pixels take arbitrary values."""
    amount = _lerp(0.002, 0.15, s)
    mask = rng.random(img.shape[:2]) < amount
    out = img.copy()
    out[mask] = rng.random((int(mask.sum()), img.shape[2])).astype(np.float32)
    return out


def add_rician_noise(img, s, rng):
    """Magnitude-MRI noise: Gaussian noise on real and imaginary parts."""
    sigma = _lerp(2, 40, s) / 255.0
    n1 = rng.normal(0, sigma, img.shape[:2] + (1,)).astype(np.float32)
    n2 = rng.normal(0, sigma, img.shape[:2] + (1,)).astype(np.float32)
    return _clip(np.sqrt((img + n1) ** 2 + n2 ** 2))


def add_stripe_noise(img, s, rng):
    """Row / column fixed-pattern noise, or smooth sinusoidal banding."""
    h, w = img.shape[:2]
    rows = rng.random() < 0.5
    n = h if rows else w
    if rng.random() < 0.7:
        line = rng.normal(0, _lerp(1, 25, s) / 255.0, n)
    else:
        line = _lerp(0.01, 0.08, s) * np.sin(np.linspace(0, rng.uniform(6, 40) * np.pi, n) + rng.uniform(0, 6.28))
    line = line.astype(np.float32)
    offset = line[:, None, None] if rows else line[None, :, None]
    return _clip(img + offset)


def apply_resize_blur(img, s, rng):
    """Down- then up-sampling: the softness of an image that was once smaller."""
    h, w = img.shape[:2]
    factor = _lerp(1.2, 4.0, s)
    down = [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC][rng.integers(3)]
    up = [cv2.INTER_LINEAR, cv2.INTER_CUBIC][rng.integers(2)]
    small = cv2.resize(img, (max(8, int(w / factor)), max(8, int(h / factor))), interpolation=down)
    return _clip(cv2.resize(small, (w, h), interpolation=up))


def apply_webp_compression(img, s, rng):
    try:
        return _pil_roundtrip(img, "WEBP", quality=int(round(_lerp(90, 5, s))))
    except Exception:          # Pillow built without WebP: fall back to JPEG
        return apply_jpeg_compression(img, s, rng)


def apply_quantization(img, s, rng):
    """Posterisation: fewer intensity levels (6 bits down to 3)."""
    levels = 2 ** int(round(_lerp(6, 3, s))) - 1
    return _clip(np.round(img * levels) / levels)


def add_small_artifacts(img, s, rng):
    """Scratches, dead-pixel clusters and tiny blocks ("small artifacts" in the brief)."""
    h, w = img.shape[:2]
    out = np.ascontiguousarray(img.copy())
    for _ in range(int(round(_lerp(1, 24, s)))):
        kind = rng.integers(3)
        colour = [(1.0,) * 3, (0.0,) * 3, tuple(float(v) for v in rng.random(3))][rng.integers(3)]
        x, y = int(rng.integers(w)), int(rng.integers(h))
        if kind == 0:
            ang, length = rng.uniform(0, 2 * np.pi), rng.uniform(6, 90)
            end = (int(x + length * np.cos(ang)), int(y + length * np.sin(ang)))
            cv2.line(out, (x, y), end, colour, 1, lineType=cv2.LINE_AA)
        elif kind == 1:
            cv2.circle(out, (x, y), int(rng.integers(1, 3)), colour, -1)
        else:
            bw, bh = int(rng.integers(2, 9)), int(rng.integers(2, 9))
            out[y:y + bh, x:x + bw] = colour
    return _clip(out)


# ------------------------- unseen families (benchmark only) ------------------------- #

def add_uniform_noise(img, s, rng):
    a = _lerp(4, 80, s) / 255.0
    return _clip(img + rng.uniform(-a, a, img.shape).astype(np.float32))


def apply_pixelate(img, s, rng):
    h, w = img.shape[:2]
    factor = _lerp(1.5, 5.0, s)
    small = cv2.resize(img, (max(8, int(w / factor)), max(8, int(h / factor))), interpolation=cv2.INTER_AREA)
    return _clip(cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST))


def add_periodic_noise(img, s, rng):
    """A diagonal 2D sinusoid (interference / moire-like pattern)."""
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    freq, ang = rng.uniform(0.03, 0.25), rng.uniform(0, np.pi)
    wave = np.sin(2 * np.pi * freq * (x * np.cos(ang) + y * np.sin(ang)) + rng.uniform(0, 6.28))
    return _clip(img + _lerp(0.01, 0.12, s) * wave[..., None])


# ------------------------- registry ------------------------- #

# name -> (function, group, position in the physical chain)
FAMILIES = {
    "chromatic_aberration": (apply_chromatic_aberration, "colour", 0),
    "gaussian_blur":        (apply_gaussian_blur,        "blur", 1),
    "motion_blur":          (apply_motion_blur,          "blur", 1),
    "defocus_blur":         (apply_defocus_blur,         "blur", 1),
    "resize_blur":          (apply_resize_blur,          "blur", 1),
    "pixelate":             (apply_pixelate,             "blur", 1),
    "poisson_noise":        (add_poisson_noise,          "noise", 2),
    "speckle_noise":        (add_speckle_noise,          "noise", 2),
    "gaussian_noise":       (add_gaussian_noise,         "noise", 3),
    "correlated_noise":     (add_correlated_noise,       "noise", 3),
    "rician_noise":         (add_rician_noise,           "noise", 3),
    "uniform_noise":        (add_uniform_noise,          "noise", 3),
    "stripe_noise":         (add_stripe_noise,           "noise", 3),
    "periodic_noise":       (add_periodic_noise,         "noise", 3),
    "salt_pepper":          (add_salt_and_pepper,        "impulse", 4),
    "random_impulse":       (add_random_impulse,         "impulse", 4),
    "small_artifacts":      (add_small_artifacts,        "artifact", 4),
    "quantization":         (apply_quantization,         "compression_like", 5),
    "jpeg":                 (apply_jpeg_compression,     "compression", 6),
    "webp":                 (apply_webp_compression,     "compression", 6),
}

LISTED_FAMILIES = ["gaussian_noise", "poisson_noise", "salt_pepper", "speckle_noise",
                   "gaussian_blur", "motion_blur", "defocus_blur", "jpeg",
                   "chromatic_aberration"]
EXTRA_FAMILIES = ["correlated_noise", "random_impulse", "rician_noise", "stripe_noise",
                  "resize_blur", "webp", "quantization", "small_artifacts"]
UNSEEN_FAMILIES = ["uniform_noise", "pixelate", "periodic_noise"]

TRAIN_FAMILIES = LISTED_FAMILIES + EXTRA_FAMILIES
NUM_TRAIN_FAMILIES = len(TRAIN_FAMILIES)

# At most this many families from one group in a single chain (two different
# blurs or two codecs stacked just destroy the image without teaching anything).
_GROUP_LIMIT = {"blur": 1, "compression": 1, "noise": 2, "impulse": 1}


def apply_recipe(img, recipe, rng, shuffle=False):
    """Applies [(family_name, severity), ...] and returns the uint8-quantised result."""
    steps = list(recipe)
    if shuffle:
        steps = [steps[i] for i in rng.permutation(len(steps))]
    else:
        steps.sort(key=lambda step: FAMILIES[step[0]][2])
    out = img
    for name, s in steps:
        out = FAMILIES[name][0](out, s, rng)
    return quantize_uint8(out)


class RandomDegradation:
    """
    Draws one random degradation chain per call.

    num_ops_probs: probability of a chain of 0, 1, 2, 3, 4 families. Zero
                   families (a clean, merely quantised input) teaches the
                   model to leave good images alone.
    listed_weight: share of draws taken from the brief's nine families; the
                   rest come from EXTRA_FAMILIES.
    shuffle_prob:  fraction of chains applied in random instead of physical order.
    """

    def __init__(self, num_ops_probs=(0.03, 0.34, 0.39, 0.19, 0.05),
                 listed_weight=0.75, shuffle_prob=0.2):
        self.num_ops_probs = np.asarray(num_ops_probs, dtype=np.float64)
        self.num_ops_probs /= self.num_ops_probs.sum()
        self.shuffle_prob = shuffle_prob
        w = np.array([listed_weight / len(LISTED_FAMILIES)] * len(LISTED_FAMILIES)
                     + [(1 - listed_weight) / len(EXTRA_FAMILIES)] * len(EXTRA_FAMILIES))
        self.family_probs = w / w.sum()

    def sample_recipe(self, rng):
        k = int(rng.choice(len(self.num_ops_probs), p=self.num_ops_probs))
        chosen, group_count = [], {}
        for idx in rng.choice(NUM_TRAIN_FAMILIES, size=NUM_TRAIN_FAMILIES, replace=False,
                              p=self.family_probs):
            if len(chosen) == k:
                break
            name = TRAIN_FAMILIES[idx]
            group = FAMILIES[name][1]
            if group_count.get(group, 0) >= _GROUP_LIMIT.get(group, 1):
                continue
            group_count[group] = group_count.get(group, 0) + 1
            chosen.append(name)
        # Longer chains use gentler individual severities, so a 3-4 family mix
        # is still a restorable image rather than noise.
        return [(name, float(rng.beta(1.2, 1.5 + 0.6 * (len(chosen) - 1)))) for name in chosen]

    def __call__(self, img, rng=None, return_label=False):
        rng = rng if rng is not None else np.random.default_rng()
        recipe = self.sample_recipe(rng)
        out = apply_recipe(img, recipe, rng, shuffle=rng.random() < self.shuffle_prob)
        if not return_label:
            return out
        # Per-family target: 0 = absent, 0.25..1 = present at that severity.
        label = np.zeros(NUM_TRAIN_FAMILIES, dtype=np.float32)
        for name, s in recipe:
            label[TRAIN_FAMILIES.index(name)] = 0.25 + 0.75 * s
        return out, label


# ------------------------- fixed benchmark definition ------------------------- #
# (case name, group, families). The benchmark applies each case to a fixed set
# of held-out images at evenly spread severities, so every architecture is
# scored on exactly the same corrupted images, broken down by case.
BENCHMARK_CASES = (
    [(name, "single", [name]) for name in LISTED_FAMILIES]
    + [(name, "extra", [name]) for name in ("correlated_noise", "resize_blur", "small_artifacts")]
    + [("+".join(f), "pair", list(f)) for f in (
        ("gaussian_blur", "gaussian_noise"), ("motion_blur", "gaussian_noise"),
        ("defocus_blur", "poisson_noise"), ("gaussian_noise", "jpeg"),
        ("gaussian_blur", "jpeg"), ("motion_blur", "jpeg"),
        ("gaussian_blur", "salt_pepper"), ("defocus_blur", "speckle_noise"),
        ("poisson_noise", "jpeg"), ("chromatic_aberration", "gaussian_noise"),
        ("gaussian_noise", "salt_pepper"), ("chromatic_aberration", "defocus_blur"))]
    + [("+".join(f), "triple", list(f)) for f in (
        ("gaussian_blur", "gaussian_noise", "jpeg"), ("motion_blur", "poisson_noise", "jpeg"),
        ("defocus_blur", "speckle_noise", "salt_pepper"),
        ("chromatic_aberration", "gaussian_blur", "gaussian_noise"),
        ("motion_blur", "salt_pepper", "jpeg"))]
    + [(name, "unseen", [name]) for name in UNSEEN_FAMILIES]
    + [("pixelate+uniform_noise", "unseen", ["pixelate", "uniform_noise"])]
)
