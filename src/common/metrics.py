"""
Evaluation metrics matching the competition's exact scoring axes
(Section 5 of the brief):
  - PSNR  (skimage.metrics.peak_signal_noise_ratio)
  - SSIM  (skimage.metrics.structural_similarity)
  - DISTS (DISTS-pytorch, official implementation)
  - FLOPs (fvcore, fixed input (1, 3, 256, 256))

All full-reference metrics are computed on uint8 [0, 255] images, matching
the brief's evaluation notes exactly, so your local validation numbers are
directly comparable to how the competition will score you.
"""
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

_dists_model = None


def _get_dists_model(device):
    global _dists_model
    if _dists_model is None:
        try:
            from DISTS_pytorch import DISTS
            _dists_model = DISTS().to(device)
            _dists_model.eval()
        except ImportError:
            print("[metrics] WARNING: DISTS_pytorch not installed. "
                  "Run `pip install DISTS-pytorch` to enable the DISTS metric. "
                  "Returning NaN for DISTS in the meantime.")
            _dists_model = False
        except Exception as e:
            # Covers offline sandboxes / flaky networks where DISTS's VGG16
            # backbone weights can't be downloaded right now (e.g. HTTP 403/timeout).
            # Never let this take down a training run -- just skip DISTS.
            print(f"[metrics] WARNING: could not initialize DISTS model ({e}). "
                  f"This usually means its pretrained VGG16 weights couldn't be "
                  f"downloaded (check internet access). Returning NaN for DISTS.")
            _dists_model = False
    return _dists_model


def tensor_to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """(C, H, W) float tensor in [0, 1] -> (H, W, C) uint8 numpy array."""
    img = img_tensor.detach().clamp(0, 1).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return (img * 255.0).round().astype(np.uint8)


def compute_psnr(pred_uint8: np.ndarray, target_uint8: np.ndarray) -> float:
    return float(sk_psnr(target_uint8, pred_uint8, data_range=255))


def compute_ssim(pred_uint8: np.ndarray, target_uint8: np.ndarray) -> float:
    return float(sk_ssim(target_uint8, pred_uint8, data_range=255, channel_axis=-1))


@torch.no_grad()
def compute_dists_batch(pred_batch: torch.Tensor, target_batch: torch.Tensor, device) -> float:
    """pred_batch/target_batch: (B, C, H, W) float in [0, 1]."""
    model = _get_dists_model(device)
    if model is False:
        return float("nan")
    pred_batch = pred_batch.to(device)
    target_batch = target_batch.to(device)
    score = model(pred_batch, target_batch)
    return float(score.mean().item())


@torch.no_grad()
def evaluate_batch(pred_batch: torch.Tensor, target_batch: torch.Tensor, device):
    """
    Computes PSNR/SSIM per-image (matching the brief's per-image uint8
    protocol) and DISTS on the whole batch (DISTS expects [0,1] float
    tensors, no uint8 round-trip needed since we don't quantize for it).
    Returns a dict of mean values across the batch.
    """
    b = pred_batch.shape[0]
    psnrs, ssims = [], []
    for i in range(b):
        pred_u8 = tensor_to_uint8(pred_batch[i])
        tgt_u8 = tensor_to_uint8(target_batch[i])
        psnrs.append(compute_psnr(pred_u8, tgt_u8))
        ssims.append(compute_ssim(pred_u8, tgt_u8))

    dists_val = compute_dists_batch(pred_batch, target_batch, device)

    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "dists": dists_val,
    }


def compute_flops_and_params(model, input_size=(1, 3, 256, 256), device="cpu"):
    """
    Measures FLOPs the same way the competition does (Section 5: fixed
    input tensor of shape (1, 3, 256, 256), via fvcore/ptflops). Tries
    fvcore first (used as the brief's reference implementation), falls
    back to ptflops if fvcore is unavailable.
    """
    model = model.to(device).eval()
    dummy = torch.randn(*input_size, device=device)

    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
        with torch.no_grad():
            flops = FlopCountAnalysis(model, dummy)
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            total_flops = flops.total()
        total_params = parameter_count(model)[""]
        return {"flops": float(total_flops), "gflops": float(total_flops) / 1e9,
                "params": int(total_params)}
    except Exception as e:
        print(f"[metrics] fvcore FLOPs counting failed ({e}), trying ptflops...")

    try:
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(
            model, input_size[1:], as_strings=False,
            print_per_layer_stat=False, verbose=False,
        )
        flops = macs * 2  # MACs -> FLOPs
        return {"flops": float(flops), "gflops": float(flops) / 1e9, "params": int(params)}
    except Exception as e:
        print(f"[metrics] ptflops also failed ({e}). Returning parameter count only.")
        n_params = sum(p.numel() for p in model.parameters())
        return {"flops": float("nan"), "gflops": float("nan"), "params": int(n_params)}
