"""
Conditional GAN training for round 4 (a20, a21).

Generator: src/common/gennet.py::GenNet. Discriminator (training only, so it
adds no inference FLOPs): the U-Net discriminator with spectral
normalisation from Real-ESRGAN (Wang et al. 2021), which scores every pixel
real/fake. It is conditional (pix2pix): it always sees the corrupted input
next to the candidate, so it judges "the right restoration of this input".

disc_input:
  rgb     - candidate and corrupted image, 6 channels at 256px.
  wavelet - the three high-frequency Haar bands of each, 18 channels at
            128px (Korkmaz et al., CVPR 2024): noise, blur and compression
            artifacts live in those bands. Colour/brightness errors are left
            to the reconstruction losses.

Generator loss = composite loss (Charbonnier + FFT + SSIM + DISTS)
                 + wavelet_weight * L1 on Haar bands
                 + adv_weight * hinge adversarial term (ramped in from step 0).
Discriminator loss = hinge + lazy R1 gradient penalty.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .bundle import RestorationBundle, _grad_scaler
from .layers import haar_dwt


def wavelet_l1(pred, target):
    """L1 distance between the Haar sub-bands of two image batches."""
    return (haar_dwt(pred.float()) - haar_dwt(target.float())).abs().mean()


class UNetDiscriminatorSN(nn.Module):
    def __init__(self, in_ch=6, feat=32):
        super().__init__()
        sn = spectral_norm
        self.conv0 = nn.Conv2d(in_ch, feat, 3, 1, 1)
        self.conv1 = sn(nn.Conv2d(feat, feat * 2, 4, 2, 1, bias=False))
        self.conv2 = sn(nn.Conv2d(feat * 2, feat * 4, 4, 2, 1, bias=False))
        self.conv3 = sn(nn.Conv2d(feat * 4, feat * 8, 4, 2, 1, bias=False))
        self.conv4 = sn(nn.Conv2d(feat * 8, feat * 4, 3, 1, 1, bias=False))
        self.conv5 = sn(nn.Conv2d(feat * 4, feat * 2, 3, 1, 1, bias=False))
        self.conv6 = sn(nn.Conv2d(feat * 2, feat, 3, 1, 1, bias=False))
        self.conv7 = sn(nn.Conv2d(feat, feat, 3, 1, 1, bias=False))
        self.conv8 = sn(nn.Conv2d(feat, feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(feat, 1, 3, 1, 1)

    def forward(self, x):
        act = lambda t: F.leaky_relu(t, 0.2)
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        x0 = act(self.conv0(x))
        x1 = act(self.conv1(x0))
        x2 = act(self.conv2(x1))
        x3 = act(self.conv3(x2))
        x4 = act(self.conv4(up(x3))) + x2
        x5 = act(self.conv5(up(x4))) + x1
        x6 = act(self.conv6(up(x5))) + x0
        out = act(self.conv7(x6))
        out = act(self.conv8(out))
        return self.conv9(out)


class GANBundle(RestorationBundle):
    def __init__(self, cfg, device, model, name="gan"):
        super().__init__(cfg, device, model, name=name)
        t = cfg["train"]
        g = dict(t.get("gan") or {})
        self.adv_weight = g.get("adv_weight", 0.005)
        self.ramp_steps = max(1, g.get("ramp_steps", 5000))
        self.r1_gamma = g.get("r1_gamma", 1.0)
        self.r1_every = g.get("r1_every", 16)
        self.disc_input = g.get("disc_input", "rgb")
        self.wavelet_weight = t.get("wavelet_weight", 0.0)
        in_ch = 18 if self.disc_input == "wavelet" else 6
        self.disc = UNetDiscriminatorSN(in_ch, g.get("disc_feat", 32)).to(device)
        self.opt_d = torch.optim.AdamW(self.disc.parameters(), lr=self.base_lr,
                                       betas=(0.9, 0.99), weight_decay=0.0)
        self.scaler_d = _grad_scaler(self.use_amp)
        print(f"[{name}] discriminator: {self.disc_input}, "
              f"{sum(p.numel() for p in self.disc.parameters()):,} parameters "
              f"(training only)")

    def _d_in(self, image, corrupted):
        image = image.to(corrupted.dtype) if image.dtype != corrupted.dtype else image
        if self.disc_input == "wavelet":
            return torch.cat([haar_dwt(image)[:, 3:], haar_dwt(corrupted)[:, 3:]], dim=1)
        return torch.cat([image, corrupted], dim=1)

    def train_step(self, corrupted, clean, labels=None):
        self.model.train()
        self.disc.train()
        lr = self._lr_at(self.step)
        for opt in (self.optimizer, self.opt_d):
            for group in opt.param_groups:
                group["lr"] = lr
        if self.on_cuda:
            corrupted = corrupted.contiguous(memory_format=torch.channels_last)
        amp = dict(device_type="cuda", dtype=torch.float16, enabled=self.use_amp)

        with torch.autocast(**amp):
            pred = self.model(corrupted)
        fake = pred.detach().float().clamp(0, 1)

        # ---- discriminator ----
        self.opt_d.zero_grad(set_to_none=True)
        with torch.autocast(**amp):
            d_real = self.disc(self._d_in(clean, corrupted))
            d_fake = self.disc(self._d_in(fake, corrupted))
        loss_d = F.relu(1.0 - d_real.float()).mean() + F.relu(1.0 + d_fake.float()).mean()
        if self.r1_gamma > 0 and self.step % self.r1_every == 0:
            real_in = self._d_in(clean, corrupted).float().detach().requires_grad_(True)
            with torch.autocast(device_type="cuda", enabled=False):
                score = self.disc(real_in).float().mean(dim=(1, 2, 3)).sum()
            grad, = torch.autograd.grad(score, real_in, create_graph=True)
            loss_d = loss_d + 0.5 * self.r1_gamma * self.r1_every * \
                grad.pow(2).sum(dim=(1, 2, 3)).mean()
        if torch.isfinite(loss_d):
            self.scaler_d.scale(loss_d).backward()
            self.scaler_d.step(self.opt_d)
            self.scaler_d.update()

        # ---- generator ----
        self.optimizer.zero_grad(set_to_none=True)
        for p in self.disc.parameters():
            p.requires_grad_(False)
        try:
            with torch.autocast(**amp):
                adv = -self.disc(self._d_in(pred, corrupted)).float().mean()
            loss, parts = self.criterion(pred, clean, self.step / self.total_steps)
            if self.wavelet_weight > 0:
                parts["wave"] = wavelet_l1(pred, clean)
                loss = loss + self.wavelet_weight * parts["wave"]
                parts["wave"] = parts["wave"].item()
            ramp = min(1.0, (self.step + 1) / self.ramp_steps)
            loss = loss + self.adv_weight * ramp * adv
            if not torch.isfinite(loss):
                return self._skip_step()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       max_norm=self.grad_clip)
            if not torch.isfinite(grad_norm):
                self.scaler.update()
                return self._skip_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        finally:
            for p in self.disc.parameters():
                p.requires_grad_(True)
        self.step += 1
        self._update_ema()
        return {"loss": loss.item(), **parts, "gen_loss": adv.item(),
                "disc_loss": loss_d.item()}

    def state_dict(self):
        state = super().state_dict()
        state.update(disc=self.disc.state_dict(), opt_d=self.opt_d.state_dict())
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if "disc" in state:
            self.disc.load_state_dict(state["disc"])
        if "opt_d" in state:
            self.opt_d.load_state_dict(state["opt_d"])
