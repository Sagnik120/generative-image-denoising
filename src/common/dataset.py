"""
Dataset acquisition + paired (corrupted, clean) dataset for training.

Dataset choice (see docs/ for the full rationale):
  - DIV2K_train_HR   (800 high-quality diverse natural images, 2K resolution)
  - DIV2K_valid_HR   (100 images, used as a natural-image validation split)
  - BSDS500          (500 diverse natural images, classic restoration benchmark)
These three together give strong content diversity (people, textures,
architecture, nature, animals) at high enough resolution to densely crop
many distinct 256x256 patches per image -- important because your
corruption pipeline is randomized on-the-fly, so more source pixels means
more effective training variety.

Optional extra-domain packs (low-light / medical) can be dropped into
`data/extra/<domain_name>/*.png` and will be automatically picked up --
see docs/dataset_notes.md for where to source license-appropriate low-light
and medical (X-ray / MRI) images. The pipeline works fully without them,
but adding even a modest slice measurably helps cross-domain transfer,
per the brief's Section 4 (held-out set spans natural / low-light / medical).

Everything is downloaded ONCE into a persistent directory (intended to be
a Google Drive path when run from Colab) and cached: if the files already
exist, nothing is re-downloaded, so a Colab session restart does not cost
you another multi-GB download.
"""
import os
import zipfile
import tarfile
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .degradations import RandomDegradation

DIV2K_TRAIN_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DIV2K_VALID_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
BSDS500_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSR_bsds500.tgz"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _download_with_resume(url: str, dest_path: Path, chunk_size: int = 1 << 20):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    resume_from = tmp_path.stat().st_size if tmp_path.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + resume_from
        mode = "ab" if resume_from else "wb"
        with open(tmp_path, mode) as f, tqdm(
            total=total, initial=resume_from, unit="B", unit_scale=True,
            desc=f"Downloading {dest_path.name}"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    tmp_path.rename(dest_path)


def _extract_archive(archive_path: Path, extract_to: Path):
    extract_to.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_to)
    elif archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix == ".tgz":
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive type: {archive_path}")


def _has_images(folder: Path, min_count: int = 1) -> bool:
    if not folder.exists():
        return False
    count = 0
    for p in folder.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            count += 1
            if count >= min_count:
                return True
    return False


def ensure_datasets(data_root: str, download_bsds: bool = True) -> dict:
    """
    Ensures DIV2K (train + valid) and optionally BSDS500 exist under
    `data_root`. Downloads are cached: if the extracted images are already
    present, this function does nothing (safe to call every notebook run).

    `data_root` should point at a PERSISTENT location, e.g.
    "/content/drive/MyDrive/denoising_competition/data" when run in Colab,
    so that a runtime disconnect never forces a re-download.

    Returns a dict of {name: Path} for each dataset's image folder.
    """
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {}

    # ---- DIV2K train ----
    div2k_train_dir = root / "DIV2K_train_HR"
    if not _has_images(div2k_train_dir, min_count=700):
        archive = root / "DIV2K_train_HR.zip"
        if not archive.exists():
            print("[dataset] Downloading DIV2K_train_HR (~3.3GB, one-time)...")
            _download_with_resume(DIV2K_TRAIN_URL, archive)
        print("[dataset] Extracting DIV2K_train_HR...")
        _extract_archive(archive, root)
    else:
        print("[dataset] DIV2K_train_HR already present, skipping download.")
    paths["div2k_train"] = div2k_train_dir

    # ---- DIV2K valid ----
    div2k_valid_dir = root / "DIV2K_valid_HR"
    if not _has_images(div2k_valid_dir, min_count=80):
        archive = root / "DIV2K_valid_HR.zip"
        if not archive.exists():
            print("[dataset] Downloading DIV2K_valid_HR (~450MB, one-time)...")
            _download_with_resume(DIV2K_VALID_URL, archive)
        print("[dataset] Extracting DIV2K_valid_HR...")
        _extract_archive(archive, root)
    else:
        print("[dataset] DIV2K_valid_HR already present, skipping download.")
    paths["div2k_valid"] = div2k_valid_dir

    # ---- BSDS500 (optional but recommended: adds scene diversity) ----
    if download_bsds:
        bsds_dir = root / "BSR" / "BSDS500" / "data" / "images"
        if not _has_images(bsds_dir, min_count=300):
            archive = root / "BSR_bsds500.tgz"
            if not archive.exists():
                try:
                    print("[dataset] Downloading BSDS500 (~70MB, one-time)...")
                    _download_with_resume(BSDS500_URL, archive)
                    print("[dataset] Extracting BSDS500...")
                    _extract_archive(archive, root)
                except Exception as e:
                    print(f"[dataset] WARNING: BSDS500 download failed ({e}). "
                          f"Continuing with DIV2K only.")
        else:
            print("[dataset] BSDS500 already present, skipping download.")
        paths["bsds500"] = bsds_dir

    # ---- Optional user-provided extra domains (low-light / medical) ----
    extra_dir = root / "extra"
    if extra_dir.exists():
        for sub in sorted(p for p in extra_dir.iterdir() if p.is_dir()):
            if _has_images(sub, min_count=1):
                paths[f"extra_{sub.name}"] = sub
                print(f"[dataset] Found extra domain pack: {sub.name} "
                      f"({sum(1 for _ in sub.rglob('*')) } files)")

    return paths


def collect_image_paths(dataset_paths: dict, exclude_keys: Optional[List[str]] = None) -> List[str]:
    exclude_keys = exclude_keys or []
    all_paths = []
    for key, folder in dataset_paths.items():
        if key in exclude_keys:
            continue
        folder = Path(folder)
        if not folder.exists():
            continue
        for p in folder.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                all_paths.append(str(p))
    return sorted(all_paths)


class DenoisingDataset(Dataset):
    """
    Loads a clean image, takes a random 256x256 crop, and applies the
    randomized on-the-fly degradation pipeline to produce the corrupted
    counterpart. Returns (corrupted, clean) as float32 CHW tensors in [0, 1].
    """

    def __init__(self, image_paths: List[str], patch_size: int = 256,
                 augment: bool = True, degrade: bool = True,
                 min_ops: int = 1, max_ops: int = 3, seed: Optional[int] = None):
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.augment = augment
        self.degrade_fn = RandomDegradation(min_ops=min_ops, max_ops=max_ops) if degrade else None
        if seed is not None:
            random.seed(seed)

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            # Corrupt/unreadable file safety net -- fall back to a neighbor.
            return self._load_image(random.choice(self.image_paths))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    def _random_crop(self, img: np.ndarray) -> np.ndarray:
        h, w, _ = img.shape
        ps = self.patch_size
        if h < ps or w < ps:
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            h, w, _ = img.shape
        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        return img[top:top + ps, left:left + ps, :]

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            img = np.fliplr(img)
        if random.random() < 0.5:
            img = np.flipud(img)
        k = random.randint(0, 3)
        if k:
            img = np.rot90(img, k)
        return np.ascontiguousarray(img)

    def __getitem__(self, idx):
        clean = self._load_image(self.image_paths[idx])
        clean = self._random_crop(clean)
        if self.augment:
            clean = self._augment(clean)

        if self.degrade_fn is not None:
            corrupted = self.degrade_fn(clean)
        else:
            corrupted = clean.copy()

        clean_t = torch.from_numpy(clean.transpose(2, 0, 1).copy()).float()
        corrupted_t = torch.from_numpy(corrupted.transpose(2, 0, 1).copy()).float()
        return corrupted_t, clean_t


def build_dataloaders(data_root: str, patch_size: int = 256, batch_size: int = 16,
                       num_workers: int = 4, val_fraction: float = 0.03,
                       download_bsds: bool = True, seed: int = 42):
    """
    High-level convenience function: ensures datasets are downloaded/cached,
    builds the train/val split, and returns (train_loader, val_loader).
    """
    dataset_paths = ensure_datasets(data_root, download_bsds=download_bsds)

    # DIV2K_valid is a natural held-out-style split; use it as validation,
    # and keep a small slice of the training pool out too, for extra safety.
    all_train_paths = collect_image_paths(dataset_paths, exclude_keys=["div2k_valid"])
    val_native_paths = collect_image_paths({"div2k_valid": dataset_paths.get("div2k_valid", "")})

    random.Random(seed).shuffle(all_train_paths)
    n_val_extra = int(len(all_train_paths) * val_fraction)
    val_extra_paths = all_train_paths[:n_val_extra]
    train_paths = all_train_paths[n_val_extra:]
    val_paths = val_native_paths + val_extra_paths

    print(f"[dataset] {len(train_paths)} training images, {len(val_paths)} validation images.")

    train_ds = DenoisingDataset(train_paths, patch_size=patch_size, augment=True, degrade=True)
    val_ds = DenoisingDataset(val_paths, patch_size=patch_size, augment=False, degrade=True, seed=123)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True,
                               persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=max(0, num_workers // 2), pin_memory=True,
                             persistent_workers=num_workers > 0)
    return train_loader, val_loader
