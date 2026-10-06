"""
Dataset acquisition, caching, and the training / benchmark datasets.

The held-out test set (brief Section 4) spans natural photography, low-light
imagery and medical imaging, at 256x256. Training data is therefore a MIX of
domains, sampled by weight rather than concatenated, so a 40k-image natural
set cannot drown out a 600-image ultrasound set:

  natural    DIV2K, Flickr2K, BSDS500, COCO test2017
  low-light  LOL (real low- and normal-light pairs) + synthetic darkening
  medical    NIH ChestX-ray (X-ray), IXI (brain MRI), MedMNIST OrganC (CT)
             and Breast (ultrasound)

Everything is fetched by `ensure_datasets()` (called from
scripts/prepare_data.py), which
  - tries each source's mirrors in turn, resuming partial downloads;
  - converts large or unusual formats into ready-to-load PNGs once
    (2K photos -> a 512px whole view + native 512px tiles, MRI volumes ->
    slices), so training never decodes a 2K PNG to take one crop;
  - writes `manifest.json` with the image count of every source.
Training REFUSES to start if a required view is missing or short: silently
training on less data (which is what happened in round 1, when the BSDS500
URL had gone dead) is exactly the failure this guards against.

It also builds the fixed benchmark: ~400 held-out images, each with a
pre-generated corruption saved to disk, identical for every architecture.
"""
import fcntl
import json
import os
import tarfile
import time
import zipfile
import zlib
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .degradations import (RandomDegradation, apply_recipe, quantize_uint8,
                           BENCHMARK_CASES, NUM_TRAIN_FAMILIES)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
BENCHMARK_VERSION = "v1"

# name -> mirrors [(url, archive filename)], approximate size, and the minimum
# number of items that must be present for the source to count as complete.
SOURCES = {
    "div2k_train": {"gb": 3.5, "min_items": 800, "mirrors": [
        ("https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip", "DIV2K_train_HR.zip")]},
    "div2k_valid": {"gb": 0.45, "min_items": 100, "mirrors": [
        ("https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip", "DIV2K_valid_HR.zip"),
        ("https://huggingface.co/datasets/eugenesiow/Div2k/resolve/main/data/DIV2K_valid_HR.zip", "DIV2K_valid_HR.zip")]},
    "flickr2k": {"gb": 11.6, "min_items": 2600, "mirrors": [
        ("https://huggingface.co/datasets/yangtao9009/Flickr2K/resolve/main/Flickr2K.zip", "Flickr2K.zip"),
        ("https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar", "Flickr2K.tar")]},
    "bsds500": {"gb": 0.07, "min_items": 500, "mirrors": [
        ("https://github.com/BIDS/BSDS500/archive/refs/heads/master.zip", "BSDS500.zip")]},
    "coco_test2017": {"gb": 6.6, "min_items": 40000, "mirrors": [
        ("http://images.cocodataset.org/zips/test2017.zip", "coco_test2017.zip")]},
    "coco_val2017": {"gb": 0.8, "min_items": 5000, "mirrors": [
        ("http://images.cocodataset.org/zips/val2017.zip", "coco_val2017.zip")]},
    "lol": {"gb": 0.35, "min_items": 1000, "mirrors": [
        ("https://huggingface.co/datasets/geekyrakshit/LoL-Dataset/resolve/main/lol_dataset.zip", "lol_dataset.zip")]},
    "nih_xray": {"gb": 2.0, "min_items": 4900, "mirrors": [
        ("https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz", "nih_images_001.tar.gz")]},
    "ixi_t2": {"gb": 3.9, "min_items": 500, "mirrors": [
        ("https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar", "IXI-T2.tar"),
        ("http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar", "IXI-T2.tar")]},
    "organcmnist": {"gb": 0.76, "min_items": 20000, "mirrors": [
        ("https://zenodo.org/records/10519652/files/organcmnist_224.npz", "organcmnist_224.npz")]},
    "breastmnist": {"gb": 0.03, "min_items": 700, "mirrors": [
        ("https://zenodo.org/records/10519652/files/breastmnist_224.npz", "breastmnist_224.npz")]},
}

# Training views: sampling weight and the range the image's short side is
# resized to before the 256 crop. The test set is whole images shown at 256,
# so most views are shrunk towards 256 rather than cropped at native zoom.
TRAIN_VIEWS = {
    #  view              weight  short-side range  domain
    "coco":            (0.26, (256, 320), "natural"),
    "natural_whole":   (0.22, (256, 448), "natural"),
    "natural_tiles":   (0.16, (256, 512), "natural"),
    "bsds":            (0.04, (256, 321), "natural"),
    "lol_high":        (0.02, (256, 400), "natural"),
    "lol_low":         (0.03, (256, 400), "lowlight"),
    "xray":            (0.08, (256, 384), "medical"),
    "mri":             (0.07, (256, 320), "medical"),
    "ct":              (0.03, (256, 300), "medical"),
    "ultrasound":      (0.02, (256, 300), "medical"),
}
HOLDOUT_VIEWS = ["hold_div2k", "hold_bsds", "hold_coco", "hold_lol_high", "hold_lol_low",
                 "hold_xray", "hold_mri", "hold_ct", "hold_ultrasound"]
# which source produces which views
SOURCE_VIEWS = {
    "div2k_train": ["natural_whole:div2k", "natural_tiles:div2k"],
    "flickr2k": ["natural_whole:flickr2k", "natural_tiles:flickr2k"],
    "div2k_valid": ["hold_div2k"],
    "bsds500": ["bsds", "hold_bsds"],
    "coco_test2017": ["coco"],
    "coco_val2017": ["hold_coco"],
    "lol": ["lol_high", "lol_low", "hold_lol_high", "hold_lol_low"],
    "nih_xray": ["xray", "hold_xray"],
    "ixi_t2": ["mri", "hold_mri"],
    "organcmnist": ["ct", "hold_ct"],
    "breastmnist": ["ultrasound", "hold_ultrasound"],
}
SYNTH_LOWLIGHT_PROB = 0.10     # share of natural samples darkened into low-light ones
GRAYSCALE_PROB = 0.04


# --------------------------------------------------------------------------- #
# download / extract
# --------------------------------------------------------------------------- #

def _download_with_resume(url: str, dest_path: Path, chunk_size: int = 1 << 20, retries: int = 4):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    last_error = None
    for attempt in range(retries):
        try:
            resume_from = tmp_path.stat().st_size if tmp_path.exists() else 0
            headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if resume_from and r.status_code == 416:       # already complete
                    break
                r.raise_for_status()
                if resume_from and r.status_code != 206:       # server ignored Range
                    resume_from = 0
                total = int(r.headers.get("content-length", 0)) + resume_from
                with open(tmp_path, "ab" if resume_from else "wb") as f, tqdm(
                        total=total or None, initial=resume_from, unit="B", unit_scale=True,
                        desc=f"Downloading {dest_path.name}", mininterval=5) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                if total and tmp_path.stat().st_size < total:
                    raise IOError(f"connection closed early ({tmp_path.stat().st_size}/{total} bytes)")
            break
        except Exception as e:
            last_error = e
            print(f"[dataset]   attempt {attempt + 1}/{retries} failed: {e}")
            time.sleep(5 * (attempt + 1))
    else:
        raise IOError(f"download failed after {retries} attempts: {last_error}")
    tmp_path.rename(dest_path)


def _extract_archive(archive_path: Path, extract_to: Path):
    extract_to.mkdir(parents=True, exist_ok=True)
    name = archive_path.name
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_to)
    elif name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive type: {archive_path}")


def _list_images(folder: Path) -> List[Path]:
    return sorted(p for p in Path(folder).rglob("*")
                  if p.suffix.lower() in IMG_EXTS and not p.name.startswith("."))


def _fetch_source(root: Path, name: str) -> Path:
    """Downloads (trying each mirror) and extracts one source. Returns its raw path
    (a folder, or the .npz file itself)."""
    spec = SOURCES[name]
    raw_dir = root / "raw" / name
    marker = raw_dir / ".extracted"
    is_npz = spec["mirrors"][0][1].endswith(".npz")
    if marker.exists():
        return raw_dir / spec["mirrors"][0][1] if is_npz else raw_dir

    archive, errors = None, []
    for url, filename in spec["mirrors"]:
        candidate = (raw_dir if is_npz else root / "archives") / filename
        if candidate.exists():
            archive = candidate
            break
        try:
            print(f"[dataset] {name}: downloading ~{spec['gb']} GB from {url}")
            _download_with_resume(url, candidate)
            archive = candidate
            break
        except Exception as e:
            errors.append(f"{url}: {e}")
            print(f"[dataset] {name}: mirror failed, trying the next one.")
    if archive is None:
        raise IOError(f"all mirrors failed for '{name}':\n  " + "\n  ".join(errors))

    if not is_npz:
        print(f"[dataset] {name}: extracting {archive.name} ...")
        _extract_archive(archive, raw_dir)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok")
    return archive if is_npz else raw_dir


# --------------------------------------------------------------------------- #
# per-source preparation (each returns {view: [paths relative to data root]})
# --------------------------------------------------------------------------- #

def _write_png(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def _resize_short(img: np.ndarray, short: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = short / min(h, w)
    if abs(scale - 1.0) < 1e-3:
        return img
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (max(short, round(w * scale)), max(short, round(h * scale))),
                      interpolation=interp)


def _prep_hires_one(args):
    """One 2K photo -> a whole view (short side 512) + up to 6 native 512px tiles."""
    src, whole_path, tiles_dir, stem = args
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None or min(img.shape[:2]) < 600:
        return None, []
    _write_png(Path(whole_path), _resize_short(img, 512))
    h, w = img.shape[:2]
    spots = [(y, x) for y in range(0, h - 511, 512) for x in range(0, w - 511, 512)]
    rng = np.random.default_rng(zlib.crc32(stem.encode()))
    tiles = []
    for i in rng.permutation(len(spots))[:6]:
        y, x = spots[i]
        tile = img[y:y + 512, x:x + 512]
        if tile.std() < 8:            # featureless sky / bokeh: nothing to learn from
            continue
        out = Path(tiles_dir) / f"{stem}_{len(tiles)}.png"
        _write_png(out, tile)
        tiles.append(str(out))
    return whole_path, tiles


def _prep_hires(root, raw_dir, tag, pool):
    files = _list_images(raw_dir)
    hr = [p for p in files if "HR" in str(p.relative_to(raw_dir))]
    files = hr or files
    jobs = [(str(p), str(root / "cache" / "natural_whole" / f"{tag}_{p.stem}.png"),
             str(root / "cache" / "natural_tiles"), f"{tag}_{p.stem}") for p in files]
    whole, tiles = [], []
    for w, t in tqdm(pool.imap_unordered(_prep_hires_one, jobs, chunksize=4), total=len(jobs),
                     desc=f"caching {tag}", mininterval=5):
        if w:
            whole.append(w)
            tiles.extend(t)
    return {f"natural_whole:{tag}": sorted(whole), f"natural_tiles:{tag}": sorted(tiles)}, len(whole)


def _image_short_side(path):
    from PIL import Image
    try:
        with Image.open(path) as im:
            return str(path), min(im.size)
    except Exception:
        return str(path), 0


def _prep_coco(raw_dir, view, pool, limit=None):
    files = _list_images(raw_dir)
    sized = pool.map(_image_short_side, [str(p) for p in files], chunksize=256)
    keep = sorted(p for p, short in sized if short >= 400)
    return {view: keep[:limit] if limit else keep}, len(files)


def _resize_gray_one(args):
    src, dst, size = args
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _write_png(Path(dst), _resize_short(img, size))
    return dst


def _prep_xray(root, raw_dir, pool, n_hold=80):
    files = _list_images(raw_dir)
    jobs = [(str(p), str(root / "cache" / "xray" / p.name), 512) for p in files]
    done = sorted(d for d in tqdm(pool.imap_unordered(_resize_gray_one, jobs, chunksize=16),
                                  total=len(jobs), desc="caching xray", mininterval=5) if d)
    return {"xray": done[:-n_hold], "hold_xray": done[-n_hold:]}, len(done)


def _prep_ixi_one(args):
    """One MRI volume -> central axial slices as 256x256 grayscale PNGs."""
    src, out_dir, every = args
    import nibabel as nib
    try:
        vol = np.asanyarray(nib.load(src).dataobj).astype(np.float32)
    except Exception:
        return []
    if vol.ndim != 3:
        return []
    vol = np.moveaxis(vol, int(np.argmin(vol.shape)), 2)       # slice direction last
    scale = float(np.percentile(vol, 99.7))
    if scale <= 0:
        return []
    vol = np.clip(vol / scale, 0, 1)
    stem = Path(src).name.split(".")[0]
    out = []
    for k in range(int(vol.shape[2] * 0.25), int(vol.shape[2] * 0.8), every):
        sl = np.rot90(vol[:, :, k])
        if sl.mean() < 0.03:
            continue
        sl = (sl * 255).round().astype(np.uint8)
        if sl.shape != (256, 256):
            sl = cv2.resize(sl, (256, 256), interpolation=cv2.INTER_AREA)
        dst = Path(out_dir) / f"{stem}_{k:03d}.png"
        _write_png(dst, sl)
        out.append(str(dst))
    return out


def _prep_ixi(root, raw_dir, pool, n_hold_subjects=30):
    vols = sorted(str(p) for p in Path(raw_dir).rglob("*.nii*") if not p.name.startswith("."))
    jobs = [(v, str(root / "cache" / "mri"), 5) for v in vols]
    per_subject = [sorted(s) for s in tqdm(pool.imap(_prep_ixi_one, jobs, chunksize=2),
                                           total=len(jobs), desc="caching mri", mininterval=5)]
    per_subject = [s for s in per_subject if s]
    train = [p for s in per_subject[:-n_hold_subjects] for p in s]
    hold = [p for s in per_subject[-n_hold_subjects:] for p in s[::4]]
    return {"mri": train, "hold_mri": hold}, len(per_subject)


def _prep_medmnist(root, npz_path, view, max_train=8000, max_hold=160):
    data = np.load(npz_path)
    train = np.concatenate([data["train_images"], data["val_images"]])
    hold = data["test_images"]
    rng = np.random.default_rng(0)
    out = {view: [], f"hold_{view}": []}
    for key, arr, cap in ((view, train, max_train), (f"hold_{view}", hold, max_hold)):
        for i in sorted(rng.permutation(len(arr))[:cap]):
            img = cv2.resize(arr[i], (256, 256), interpolation=cv2.INTER_CUBIC)
            dst = root / "cache" / key / f"{i:06d}.png"
            _write_png(dst, img)
            out[key].append(str(dst))
    return out, len(train) + len(hold)


def _prepare_source(root: Path, name: str, raw, pool):
    if name in ("div2k_train", "flickr2k"):
        return _prep_hires(root, raw, name.split("_")[0], pool)
    if name == "div2k_valid":
        files = _list_images(raw)
        jobs = [(str(p), str(root / "cache" / "hold_div2k" / p.name)) for p in files]
        done = [w for w, _ in pool.imap_unordered(_prep_valid_one, jobs, chunksize=4) if w]
        return {"hold_div2k": sorted(done)}, len(done)
    if name == "bsds500":
        files = _list_images(raw)
        test = [str(p) for p in files if p.parent.name == "test"]
        train = [str(p) for p in files if p.parent.name in ("train", "val")]
        return {"bsds": train, "hold_bsds": test}, len(train) + len(test)
    if name == "coco_test2017":
        return _prep_coco(raw, "coco", pool)
    if name == "coco_val2017":
        views, n = _prep_coco(raw, "hold_coco", pool)
        return {"hold_coco": views["hold_coco"][:300]}, n
    if name == "lol":
        files = _list_images(raw)
        pick = lambda split, kind: [str(p) for p in files
                                    if split in p.parts and p.parent.name == kind]
        views = {"lol_high": pick("our485", "high"), "lol_low": pick("our485", "low"),
                 "hold_lol_high": pick("eval15", "high"), "hold_lol_low": pick("eval15", "low")}
        return views, sum(len(v) for v in views.values())
    if name == "nih_xray":
        return _prep_xray(root, raw, pool)
    if name == "ixi_t2":
        return _prep_ixi(root, raw, pool)
    if name == "organcmnist":
        return _prep_medmnist(root, raw, "ct")
    if name == "breastmnist":
        return _prep_medmnist(root, raw, "ultrasound")
    raise KeyError(name)


def _prep_valid_one(args):
    src, dst = args
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        return None, []
    _write_png(Path(dst), _resize_short(img, 512))
    return dst, []


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #

def _manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def load_manifest(data_root) -> dict:
    path = _manifest_path(Path(data_root))
    if not path.exists():
        return {"sources": {}, "views": {}}
    with open(path) as f:
        return json.load(f)


def _save_manifest(root: Path, manifest: dict):
    tmp = _manifest_path(root).with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, _manifest_path(root))


def _view_list_path(root: Path, view: str) -> Path:
    return root / "lists" / f"{view.replace(':', '__')}.txt"


def read_view(data_root, view: str) -> List[str]:
    """Paths of one view. Views fed by several sources ('natural_whole:div2k', ...)
    are merged when asked for by their base name."""
    root = Path(data_root)
    paths = []
    for f in sorted((root / "lists").glob(f"{view}.txt")) + sorted((root / "lists").glob(f"{view}__*.txt")):
        paths += [str(root / line) for line in f.read_text().splitlines() if line]
    return paths


def ensure_datasets(data_root: str, only: Optional[List[str]] = None, strict: bool = True,
                    num_procs: Optional[int] = None) -> dict:
    """
    Downloads, extracts and caches every source (or just `only`), then writes
    manifest.json. Safe to re-run: finished sources are skipped. Holds a file
    lock, so two processes started together cannot corrupt the cache.

    With strict=True, raises if any requested source ends up incomplete.
    """
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    names = only or list(SOURCES)
    failures = {}

    with open(root / ".prepare.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = load_manifest(root)
        with Pool(num_procs or min(16, os.cpu_count() or 4)) as pool:
            for name in names:
                entry = manifest["sources"].get(name, {})
                if entry.get("ok") and all(_view_list_path(root, v).exists() for v in SOURCE_VIEWS[name]):
                    print(f"[dataset] {name}: already prepared ({entry['items']} items), skipping.")
                    continue
                try:
                    raw = _fetch_source(root, name)
                    views, n_items = _prepare_source(root, name, raw, pool)
                    if n_items < SOURCES[name]["min_items"]:
                        raise IOError(f"only {n_items} items found, expected at least "
                                      f"{SOURCES[name]['min_items']}")
                    for view, paths in views.items():
                        lp = _view_list_path(root, view)
                        lp.parent.mkdir(parents=True, exist_ok=True)
                        lp.write_text("\n".join(os.path.relpath(p, root) for p in paths) + "\n")
                        manifest["views"][view] = len(paths)
                    manifest["sources"][name] = {"ok": True, "items": n_items}
                    print(f"[dataset] {name}: OK ({n_items} items).")
                except Exception as e:
                    failures[name] = str(e)
                    manifest["sources"][name] = {"ok": False, "error": str(e)}
                    print(f"[dataset] {name}: FAILED -> {e}")
                _save_manifest(root, manifest)

    if failures and strict:
        raise RuntimeError(
            "These datasets are not ready:\n"
            + "\n".join(f"  - {k}: {v}" for k, v in failures.items())
            + "\nRe-run scripts/prepare_data.py (downloads resume where they stopped).")
    return manifest


def check_training_data(data_root, allow_missing: bool = False) -> Dict[str, List[str]]:
    """Loads every training view, refusing to continue if one is missing or empty."""
    views = {v: read_view(data_root, v) for v in TRAIN_VIEWS}
    missing = [v for v, paths in views.items() if not paths]
    if missing and not allow_missing:
        raise RuntimeError(
            f"Training data is incomplete; missing views: {missing}.\n"
            f"Run:  python scripts/prepare_data.py --data_root {data_root}\n"
            f"(Pass --allow_missing_data only for a deliberate smoke test.)")
    views = {v: p for v, p in views.items() if p}
    if not views:
        raise RuntimeError(f"No training data found under {data_root}.")
    return views


# --------------------------------------------------------------------------- #
# image helpers shared by training and the benchmark
# --------------------------------------------------------------------------- #

def _load_rgb(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def synth_low_light(img: np.ndarray, rng) -> np.ndarray:
    """Turns a well-lit photo into a clean low-light one (darker, optionally with a
    colour cast). Applied to the CLEAN image; corruption is then added on top."""
    out = (img ** rng.uniform(1.0, 1.5)) * rng.uniform(0.08, 0.6)
    if rng.random() < 0.5:
        out = out * rng.uniform(0.85, 1.15, size=3).astype(np.float32)
    return np.clip(out, 0, 1).astype(np.float32)


def _center_square(img: np.ndarray, size: int = 256) -> np.ndarray:
    img = _resize_short(img, size)
    h, w = img.shape[:2]
    top, left = (h - size) // 2, (w - size) // 2
    return img[top:top + size, left:left + size]


# --------------------------------------------------------------------------- #
# fixed benchmark
# --------------------------------------------------------------------------- #

def benchmark_dir(data_root) -> Path:
    return Path(data_root) / "benchmark" / BENCHMARK_VERSION


def build_benchmark(data_root, per_case=(6, 3, 3), seed: int = 2026, force: bool = False):
    """
    Builds the fixed benchmark from held-out images: for every case in
    BENCHMARK_CASES, `per_case` = (natural, low-light, medical) images at
    severities spread evenly over [0.1, 0.9]. Saves input/target PNG pairs and
    meta.json. Deterministic for a given data root.
    """
    out_dir = benchmark_dir(data_root)
    if (out_dir / "meta.json").exists() and not force:
        print(f"[benchmark] already built at {out_dir}, skipping.")
        return out_dir
    hold = {v: read_view(data_root, v) for v in HOLDOUT_VIEWS}
    empty = [v for v, p in hold.items() if not p]
    if empty:
        raise RuntimeError(f"Cannot build the benchmark; held-out views missing: {empty}")

    rng = np.random.default_rng(seed)

    def interleave(*lists):
        lists = [list(rng.permutation(l)) for l in lists]
        out = []
        while any(lists):
            for l in lists:
                if l:
                    out.append(l.pop())
        return out

    natural = interleave(hold["hold_div2k"], hold["hold_bsds"], hold["hold_coco"])
    medical = interleave(hold["hold_xray"], hold["hold_mri"], hold["hold_ct"], hold["hold_ultrasound"])
    n_nat, n_low, n_med = per_case
    need_nat = len(BENCHMARK_CASES) * n_nat
    # low-light pool: real LOL low-light shots, then darkened held-out photos
    lowlight = [(p, False) for p in hold["hold_lol_low"]] + \
               [(p, True) for p in hold["hold_lol_high"] + natural[need_nat:]]

    (out_dir / "input").mkdir(parents=True, exist_ok=True)
    (out_dir / "target").mkdir(parents=True, exist_ok=True)
    meta, cursor = [], {"natural": 0, "lowlight": 0, "medical": 0}

    def take(domain):
        pool = {"natural": natural, "lowlight": lowlight, "medical": medical}[domain]
        item = pool[cursor[domain] % len(pool)]
        cursor[domain] += 1
        return item

    for case, group, families in tqdm(BENCHMARK_CASES, desc="building benchmark"):
        domains = ["natural"] * n_nat + ["lowlight"] * n_low + ["medical"] * n_med
        severities = np.linspace(0.1, 0.9, len(domains))
        severities = severities[rng.permutation(len(domains))]
        for domain, sev in zip(domains, severities):
            item = take(domain)
            path, darken = item if domain == "lowlight" else (item, False)
            img = _load_rgb(str(path))
            if img is None:
                continue
            clean = _center_square(img).astype(np.float32) / 255.0
            if darken:
                clean = synth_low_light(clean, rng)
            clean = quantize_uint8(clean)
            scale = {1: 1.0, 2: 0.8, 3: 0.7}.get(len(families), 0.7)
            recipe = [(f, float(sev * scale)) for f in families]
            corrupted = apply_recipe(clean, recipe, rng)
            name = f"{len(meta):04d}.png"
            for sub, arr in (("target", clean), ("input", corrupted)):
                bgr = cv2.cvtColor((arr * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_dir / sub / name), bgr)
            mse = float(np.mean((corrupted - clean) ** 2))
            meta.append({"file": name, "domain": domain, "case": case, "group": group,
                         "recipe": recipe, "source": os.path.relpath(str(path), data_root),
                         "input_psnr": round(10 * np.log10(1.0 / max(mse, 1e-10)), 3)})

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[benchmark] {len(meta)} image pairs written to {out_dir}")
    return out_dir


class BenchmarkDataset(Dataset):
    """The fixed (corrupted, clean) pairs. `meta[i]` describes pair i."""

    def __init__(self, data_root):
        self.dir = benchmark_dir(data_root)
        with open(self.dir / "meta.json") as f:
            self.meta = json.load(f)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        name = self.meta[idx]["file"]
        pair = []
        for sub in ("input", "target"):
            img = _load_rgb(str(self.dir / sub / name)).astype(np.float32) / 255.0
            pair.append(torch.from_numpy(img.transpose(2, 0, 1).copy()))
        return pair[0], pair[1], idx


# --------------------------------------------------------------------------- #
# training dataset
# --------------------------------------------------------------------------- #

class MixedDomainDataset(Dataset):
    """
    Each item: pick a view by weight -> pick an image -> shrink it towards 256
    -> random 256 crop -> flips/rotations -> (maybe darken) -> random
    degradation chain. Returns (corrupted, clean, label) with `label` the
    per-family severity vector from RandomDegradation.

    The index is ignored (every item is an independent random draw), so
    `length` simply sets how many steps make an "epoch".
    """

    def __init__(self, views: Dict[str, List[str]], length: int, patch_size: int = 256,
                 view_specs: dict = None, degrade_cfg: dict = None):
        specs = view_specs or TRAIN_VIEWS
        self.names = [v for v in views if v in specs]
        self.paths = [views[v] for v in self.names]
        self.ranges = [specs[v][1] for v in self.names]
        self.domains = [specs[v][2] for v in self.names]
        w = np.array([specs[v][0] for v in self.names], dtype=np.float64)
        self.weights = w / w.sum()
        self.length = length
        self.patch_size = patch_size
        self.degrade = RandomDegradation(**(degrade_cfg or {}))
        self.rng = None

    def __len__(self):
        return self.length

    def _rng(self):
        if self.rng is None:
            info = torch.utils.data.get_worker_info()
            seed = info.seed if info is not None else int(torch.randint(0, 2 ** 31, (1,)).item())
            self.rng = np.random.default_rng(seed)
            cv2.setNumThreads(0)      # one thread per worker; avoids CPU oversubscription
        return self.rng

    def _crop(self, img, short_range, native_prob, rng):
        ps = self.patch_size
        h, w = img.shape[:2]
        scale = ps / 256.0
        lo, hi = short_range[0] * scale, min(short_range[1] * scale, min(h, w))
        if min(h, w) < ps or not (rng.random() < native_prob):
            img = _resize_short(img, int(round(rng.uniform(lo, max(lo, hi)))))
            h, w = img.shape[:2]
        top, left = rng.integers(0, h - ps + 1), rng.integers(0, w - ps + 1)
        return img[top:top + ps, left:left + ps]

    def __getitem__(self, idx):
        rng = self._rng()
        for _ in range(20):
            v = int(rng.choice(len(self.names), p=self.weights))
            img = _load_rgb(self.paths[v][int(rng.integers(len(self.paths[v])))])
            if img is not None:
                break
        else:
            raise RuntimeError("could not read any training image (20 consecutive failures)")

        native_prob = 0.6 if self.names[v] == "natural_tiles" else 0.0
        img = self._crop(img, self.ranges[v], native_prob, rng)
        if rng.random() < 0.5:
            img = img[:, ::-1]
        if rng.random() < 0.5:
            img = img[::-1]
        img = np.rot90(img, int(rng.integers(4)))
        clean = np.ascontiguousarray(img).astype(np.float32) / 255.0

        if self.domains[v] == "natural":
            if rng.random() < GRAYSCALE_PROB:
                clean = np.repeat(cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=2)
            if rng.random() < SYNTH_LOWLIGHT_PROB:
                clean = quantize_uint8(synth_low_light(clean, rng))

        corrupted, label = self.degrade(clean, rng, return_label=True)
        return (torch.from_numpy(corrupted.transpose(2, 0, 1).copy()),
                torch.from_numpy(clean.transpose(2, 0, 1).copy()),
                torch.from_numpy(label))


def build_dataloaders(data_root: str, cfg: dict, allow_missing_data: bool = False):
    """
    Returns (train_loader, val_loader). The validation loader is the fixed
    benchmark; training refuses to start without it (or without any training
    view) unless `allow_missing_data` is set for a smoke test.
    """
    d = cfg["data"]
    batch_size, num_workers = d["batch_size"], d["num_workers"]
    patch_size = d.get("patch_size", 256)
    views = check_training_data(data_root, allow_missing=allow_missing_data)
    print("[dataset] training views: " + ", ".join(f"{k}={len(v)}" for k, v in views.items()))

    train_ds = MixedDomainDataset(views, length=d.get("steps_per_epoch", 2500) * batch_size,
                                  patch_size=patch_size, degrade_cfg=d.get("degradation"))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=num_workers > 0,
                              prefetch_factor=4 if num_workers > 0 else None)

    if not (benchmark_dir(data_root) / "meta.json").exists():
        raise RuntimeError(
            f"The fixed benchmark is missing under {benchmark_dir(data_root)}.\n"
            f"Run:  python scripts/prepare_data.py --data_root {data_root}")
    val_ds = BenchmarkDataset(data_root)
    val_loader = DataLoader(val_ds, batch_size=d.get("val_batch_size", 16), shuffle=False,
                            num_workers=min(4, num_workers), pin_memory=True)
    print(f"[dataset] benchmark: {len(val_ds)} fixed image pairs.")
    return train_loader, val_loader
