"""Loss functions shared across architectures."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Smooth L1-like loss, standard in image restoration (more robust than plain L1/L2)."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


class PSNRLoss(nn.Module):
    """Directly optimizes (negative) PSNR -- occasionally useful as an auxiliary term."""

    def __init__(self, max_val: float = 1.0):
        super().__init__()
        self.max_val = max_val

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        mse = torch.clamp(mse, min=1e-10)
        psnr = 10 * torch.log10(self.max_val ** 2 / mse)
        return -psnr


def gan_loss(pred, is_real: bool):
    """Standard non-saturating BCE-with-logits GAN loss."""
    target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
    return F.binary_cross_entropy_with_logits(pred, target)


def build_reconstruction_loss(name: str = "charbonnier"):
    name = name.lower()
    if name == "l1":
        return nn.L1Loss()
    if name == "l2" or name == "mse":
        return nn.MSELoss()
    if name == "charbonnier":
        return CharbonnierLoss()
    raise ValueError(f"Unknown reconstruction loss: {name}")
