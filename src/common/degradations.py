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

def apply_gaussian_blur(img, ksize_range=(3, 15), sigma_range=(0.3, 3.5)):
    k = random.choice(range(ksize_range[0], ksize_range[1] + 1, 2))
    sigma = random.uniform(*sigma_range)
    return _clip(cv2.GaussianBlur(img, (k, k), sigma))

def _motion_blur_kernel(ksize, angle):
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return kernel

def apply_motion_blur(img, ksize_range=(5, 21), angle_range=(0, 360)):
    ksize = random.choice(range(ksize_range[0], ksize_range[1] + 1, 2))
    angle = random.uniform(*angle_range)
    kernel = _motion_blur_kernel(ksize, angle)
    return _clip(cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT))

def _disk_kernel(radius):
    size = radius * 2 + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x ** 2 + y ** 2 <= radius ** 2
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[mask] = 1.0
    kernel /= kernel.sum()
    return kernel

def apply_defocus_blur(img, radius_range=(1, 9)):
    radius = random.randint(*radius_range)
    kernel = _disk_kernel(radius)
    return _clip(cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT))

def apply_jpeg_compression(img, quality_range=(10, 75)):
    quality = random.randint(*quality_range)
    img_uint8 = (img * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    decoded = np.array(Image.open(buf).convert("RGB")).astype(np.float32) / 255.0
    return _clip(decoded)

def apply_chromatic_aberration(img, shift_range=(1, 6)):
    h, w, _ = img.shape
    shift = random.randint(*shift_range)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    r_shifted = np.roll(r, shift, axis=1)
    b_shifted = np.roll(b, -shift, axis=0)
    return _clip(np.stack([r_shifted, g, b_shifted], axis=-1))

def apply_downsample_blur(img, factor_range=(2, 4)):
    h, w, _ = img.shape
    factor = random.uniform(*factor_range)
    small = cv2.resize(img, (max(1, int(w / factor)), max(1, int(h / factor))),
                        interpolation=cv2.INTER_LINEAR)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return _clip(back)

def apply_brightness_contrast_jitter(img, brightness_range=(-0.15, 0.15), contrast_range=(0.75, 1.25)):
    brightness = random.uniform(*brightness_range)
    contrast = random.uniform(*contrast_range)
    out = (img - 0.5) * contrast + 0.5 + brightness
    return _clip(out)

def apply_sensor_banding(img, strength_range=(0.01, 0.06)):
    h, w, _ = img.shape
    strength = random.uniform(*strength_range)
    bands = np.sin(np.linspace(0, random.uniform(10, 40) * np.pi, h)) * strength
    out = img + bands[:, None, None]
    return _clip(out)
