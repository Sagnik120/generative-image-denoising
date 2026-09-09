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

_dists_model = None

def _get_dists_model(device):
    global _dists_model
    if _dists_model is None:
        try:
            from DISTS_pytorch import DISTS
            _dists_model = DISTS().to(device)
            _dists_model.eval()
        except Exception:
            _dists_model = False
    return _dists_model

@torch.no_grad()
def compute_dists_batch(pred_batch: torch.Tensor, target_batch: torch.Tensor, device) -> float:
    model = _get_dists_model(device)
    if model is False:
        return float("nan")
    pred_batch = pred_batch.to(device)
    target_batch = target_batch.to(device)
    score = model(pred_batch, target_batch)
    return float(score.mean().item())

@torch.no_grad()
def evaluate_batch(pred_batch: torch.Tensor, target_batch: torch.Tensor, device):
    b = pred_batch.shape[0]
    psnrs, ssims = [], []
    for i in range(b):
        pred_u8 = tensor_to_uint8(pred_batch[i])
        tgt_u8 = tensor_to_uint8(target_batch[i])
        psnrs.append(compute_psnr(pred_u8, tgt_u8))
        ssims.append(compute_ssim(pred_u8, tgt_u8))
    dists_val = compute_dists_batch(pred_batch, target_batch, device)
    return {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)), "dists": dists_val}
