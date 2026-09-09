"""Evaluation metrics matching competition specifications."""
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

def tensor_to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    img = img_tensor.detach().clamp(0, 1).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return (img * 255.0).round().astype(np.uint8)

def compute_psnr(pred_uint8: np.ndarray, target_uint8: np.ndarray) -> float:
    return float(sk_psnr(target_uint8, pred_uint8, data_range=255))

def compute_ssim(pred_uint8: np.ndarray, target_uint8: np.ndarray) -> float:
    return float(sk_ssim(target_uint8, pred_uint8, data_range=255, channel_axis=-1))
