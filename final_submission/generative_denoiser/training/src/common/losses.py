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


# --------------------------------------------------------------------------- #
# Losses added for the round-2 architectures (a05 onward): one term per scored
# metric, so PSNR, SSIM and DISTS are each optimised directly.
# --------------------------------------------------------------------------- #

class SSIMLoss(nn.Module):
    """1 - SSIM, computed the way the competition's reference does
    (skimage.metrics.structural_similarity defaults: 7x7 uniform window,
    sample covariance, K1=0.01, K2=0.03), so the loss matches the metric."""

    def __init__(self, win: int = 7, data_range: float = 1.0):
        super().__init__()
        self.win = win
        self.c1 = (0.01 * data_range) ** 2
        self.c2 = (0.03 * data_range) ** 2
        self.cov_norm = win * win / (win * win - 1.0)

    def forward(self, pred, target):
        pool = lambda t: F.avg_pool2d(t, self.win, stride=1)
        ux, uy = pool(pred), pool(target)
        vx = self.cov_norm * (pool(pred * pred) - ux * ux)
        vy = self.cov_norm * (pool(target * target) - uy * uy)
        vxy = self.cov_norm * (pool(pred * target) - ux * uy)
        ssim = ((2 * ux * uy + self.c1) * (2 * vxy + self.c2)) / \
               ((ux * ux + uy * uy + self.c1) * (vx + vy + self.c2))
        return 1.0 - ssim.mean()


class FFTLoss(nn.Module):
    """L1 distance between Fourier spectra: penalises missing high frequencies
    (over-smoothing) that a pixel loss alone barely notices."""

    def forward(self, pred, target):
        diff = torch.fft.rfft2(pred, norm="ortho") - torch.fft.rfft2(target, norm="ortho")
        return torch.view_as_real(diff).abs().mean()


class DISTSLoss(nn.Module):
    """The scored DISTS metric itself, used as a training loss. Its VGG backbone
    is frozen; gradients flow only into the prediction."""

    def __init__(self):
        super().__init__()
        from DISTS_pytorch import DISTS
        self.net = DISTS()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()

    def train(self, mode=True):          # never leave eval mode
        return super().train(False)

    def forward(self, pred, target):
        return self.net(pred, target, require_grad=True, batch_average=True)


class CompositeLoss(nn.Module):
    """
    charbonnier * w_pix + fft * w_fft + (1 - SSIM) * w_ssim + DISTS * w_dists

    DISTS costs a VGG16 pass, so it (a) only switches on after
    `dists_start_frac` of training, once the image is roughly right, and
    (b) is computed on the first `dists_batch` images of each batch.
    Call as loss, parts = criterion(pred, target, progress) with progress in [0, 1].
    """

    def __init__(self, cfg=None):
        super().__init__()
        cfg = dict(cfg or {})
        self.w_pix = cfg.get("charbonnier", 1.0)
        self.w_fft = cfg.get("fft", 0.05)
        self.w_ssim = cfg.get("ssim", 0.2)
        self.w_dists = cfg.get("dists", 0.1)
        self.dists_start_frac = cfg.get("dists_start_frac", 0.5)
        self.dists_batch = cfg.get("dists_batch", 4)
        self.pix = CharbonnierLoss()
        self.fft = FFTLoss()
        self.ssim = SSIMLoss()
        self.dists = None
        if self.w_dists > 0:
            try:
                self.dists = DISTSLoss()
            except Exception as e:
                print(f"[losses] WARNING: DISTS loss disabled ({e}).")

    def forward(self, pred, target, progress: float = 0.0):
        pred, target = pred.float(), target.float()
        parts = {"pix": self.pix(pred, target)}
        total = self.w_pix * parts["pix"]
        if self.w_fft > 0:
            parts["fft"] = self.fft(pred, target)
            total = total + self.w_fft * parts["fft"]
        if self.w_ssim > 0:
            parts["ssim"] = self.ssim(pred.clamp(0, 1), target)
            total = total + self.w_ssim * parts["ssim"]
        if self.dists is not None and progress >= self.dists_start_frac:
            n = self.dists_batch
            parts["dists"] = self.dists(pred[:n].clamp(0, 1), target[:n])
            total = total + self.w_dists * parts["dists"]
        return total, {k: v.item() for k, v in parts.items()}
