"""Synthetic degradation pipeline."""
import random
import io
import numpy as np
import cv2
from PIL import Image

def _clip(img):
    return np.clip(img, 0.0, 1.0).astype(np.float32)

def add_gaussian_noise(img, sigma_range=(2, 50)):
    sigma = random.uniform(*sigma_range) / 255.0
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return _clip(img + noise)

def add_poisson_noise(img, scale_range=(1.0, 12.0)):
    scale = random.uniform(*scale_range)
    vals = 2 ** np.ceil(np.log2(scale * 30))
    noisy = np.random.poisson(img * vals) / float(vals)
    return _clip(noisy)

def add_salt_and_pepper(img, amount_range=(0.001, 0.05)):
    amount = random.uniform(*amount_range)
    out = img.copy()
    h, w, c = img.shape
    n_salt = int(amount * h * w * 0.5)
    n_pepper = int(amount * h * w * 0.5)
    ys = np.random.randint(0, h, n_salt)
    xs = np.random.randint(0, w, n_salt)
    out[ys, xs, :] = 1.0
    ys = np.random.randint(0, h, n_pepper)
    xs = np.random.randint(0, w, n_pepper)
    out[ys, xs, :] = 0.0
    return _clip(out)

def add_speckle_noise(img, sigma_range=(0.02, 0.35)):
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return _clip(img + img * noise)
