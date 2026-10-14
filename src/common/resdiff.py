"""
Residual diffusion for round 4 (a22, a23), after ResShift (Yue et al.,
NeurIPS 2023).

Ordinary diffusion ends its forward process at pure noise and needs many
reverse steps (a04 did this and collapsed). Here the forward process ends at
the CORRUPTED image plus a little noise:

    x_t = x0 + eta_t * (y - x0) + kappa * sqrt(eta_t) * eps,   t = 1..T

with x0 the clean image, y the corrupted one and eta rising from 0 to 1, so
x_T ~ N(y, kappa^2). The reverse process only has to remove the residual
y - x0, which takes very few steps.

The network predicts the clean image x0 from (x_t, y, t), so the composite
image loss applies directly. The sampler starts at y + kappa * eps and takes
K reverse steps, each with fresh noise:

    x_s = (eta_s/eta_t) x_t + (1 - eta_s/eta_t) x0_hat
          + kappa * sqrt(eta_s (eta_t - eta_s) / eta_t) * eps

All K passes run inside one forward call, so a FLOPs counter applied to the
sampler reports the true total. Noise is seeded in eval mode (see gennet.py).
"""
import math

import torch
import torch.nn as nn

from .bundle import RestorationBundle
from .gan import GANBundle, wavelet_l1
from .layers import NAFBlock, sinusoidal_time_embedding
from .unet import RestorationUNet


class ResDiffNet(nn.Module):
    """One pass: (x_t, corrupted, t) -> estimate of the clean image."""

    def __init__(self, width=24, enc_blocks=(2, 2, 2, 4), middle_blocks=3,
                 dec_blocks=(1, 1, 1, 2), time_dim=64):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim * 2), nn.SiLU(),
                                      nn.Linear(time_dim * 2, time_dim))
        widths = [width * 2 ** i for i in range(len(enc_blocks) + 1)]
        # Input = corrupted image then x_t; the output is a correction added to
        # the corrupted image (the first 3 input channels).
        self.unet = RestorationUNet(lambda level, ch: NAFBlock(ch), in_ch=6, out_ch=3,
                                    widths=widths, enc_blocks=enc_blocks,
                                    middle_blocks=middle_blocks, dec_blocks=dec_blocks,
                                    cond_dim=time_dim, predict_residual=True)

    def forward(self, x_t, corrupted, t):
        emb = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim))
        return self.unet(torch.cat([corrupted, x_t], dim=1), emb.to(x_t.dtype))


class ResDiffSampler(nn.Module):
    """The inference model: draws the noise and runs all K reverse steps."""

    def __init__(self, net, timesteps=4, steps=1, kappa=0.1, eta_power=2.0, image_size=256):
        super().__init__()
        self.net = net
        self.timesteps, self.steps, self.kappa = timesteps, steps, kappa
        self.noise_scale = 1.0       # test-time noise-scale sweep only
        self.eta = [(i / timesteps) ** eta_power for i in range(timesteps + 1)]
        self.register_buffer("eval_noise", torch.zeros(timesteps, 3, image_size, image_size))
        self.reseed(0)

    def reseed(self, seed):
        g = torch.Generator().manual_seed(int(seed) * 100003 + 7)
        noise = torch.randn(self.eval_noise.shape, generator=g)
        self.eval_noise.copy_(noise.to(self.eval_noise.device, self.eval_noise.dtype))

    def levels(self):
        """Noise levels visited, e.g. T=4: K=1 -> [4], K=2 -> [4, 2], K=4 -> [4, 3, 2, 1]."""
        k = max(1, min(self.steps, self.timesteps))
        return [max(1, round(self.timesteps * (k - i) / k)) for i in range(k)]

    def time_input(self, like, level):
        return like.new_full((like.shape[0],), level * 1000.0 / self.timesteps)

    def _noise(self, like, i):
        if self.training or self.eval_noise.shape[-2:] != like.shape[-2:]:
            return torch.randn_like(like)
        return self.noise_scale * self.eval_noise[i:i + 1].to(like.dtype)

    def forward(self, corrupted):
        levels = self.levels()
        x = corrupted + self.kappa * self._noise(corrupted, 0)
        estimate = x
        for i, t in enumerate(levels):
            estimate = self.net(x, corrupted, self.time_input(corrupted, t))
            s = levels[i + 1] if i + 1 < len(levels) else 0
            if s > 0:
                r = self.eta[s] / self.eta[t]
                sigma = self.kappa * math.sqrt(self.eta[s] * (self.eta[t] - self.eta[s]) / self.eta[t])
                x = r * x + (1.0 - r) * estimate + sigma * self._noise(corrupted, i + 1)
        return estimate


class ResDiffBundle(RestorationBundle):
    def __init__(self, cfg, device, model, name="resdiff"):
        super().__init__(cfg, device, model, name=name)
        t = cfg["train"]
        self.self_pred_prob = t.get("self_pred_prob", 0.5)
        self.last_level_prob = t.get("last_level_prob", 0.5)
        self.wavelet_weight = t.get("wavelet_weight", 0.0)
        self.samples = 1

    def forward_train(self, corrupted, clean, labels):
        sampler = self.model
        net, T, kappa = sampler.net, sampler.timesteps, sampler.kappa
        b, dev = clean.shape[0], clean.device
        # A share of the samples sits at the last level (where sampling always
        # starts); the rest are spread over all levels. With several noise draws
        # per image (round 5) all copies of one image share a level.
        n = b // self.samples
        level = torch.where(torch.rand(n, device=dev) < self.last_level_prob,
                            torch.full((n,), T, device=dev),
                            torch.randint(1, T + 1, (n,), device=dev)).repeat(self.samples)
        source = clean
        # Multi-step samplers: half the batches build x_t from the network's own
        # first estimate, which is what later steps actually receive.
        if sampler.steps > 1 and torch.rand(()).item() < self.self_pred_prob:
            with torch.no_grad():
                start = corrupted + kappa * torch.randn_like(corrupted)
                source = net(start, corrupted, sampler.time_input(corrupted, T)).float().clamp(0, 1)
        eta = torch.tensor(sampler.eta, device=dev, dtype=clean.dtype)[level].view(-1, 1, 1, 1)
        x_t = source + eta * (corrupted - source) + kappa * eta.sqrt() * torch.randn_like(clean)
        if self.on_cuda:
            x_t = x_t.contiguous(memory_format=torch.channels_last)
        pred = net(x_t, corrupted, level.float() * 1000.0 / T)
        extra, logs = None, {}
        if self.wavelet_weight > 0:
            wave = wavelet_l1(pred, clean)
            extra, logs = self.wavelet_weight * wave, {"wave": wave.item()}
        return pred, extra, logs


class ResDiffGANBundle(GANBundle):
    """Round 5 (a26): one-step residual diffusion + GAN (Denoising Diffusion
    GANs, Xiao et al. 2022). The generator pass is the residual-diffusion
    training pass (random level, clean-image estimate); GANBundle adds the
    discriminator, the adversarial term and the two-sample recipe."""

    def __init__(self, cfg, device, model, name="resdiff_gan"):
        super().__init__(cfg, device, model, name=name)
        t = cfg["train"]
        self.self_pred_prob = t.get("self_pred_prob", 0.5)
        self.last_level_prob = t.get("last_level_prob", 0.5)

    def generate(self, corrupted, clean):
        wavelet, self.wavelet_weight = self.wavelet_weight, 0.0   # GANBundle applies it
        try:
            pred, _, _ = ResDiffBundle.forward_train(self, corrupted, clean, None)
        finally:
            self.wavelet_weight = wavelet
        return pred


def build_sampler(model_cfg):
    m = dict(model_cfg or {})
    net = ResDiffNet(width=m.get("width", 24),
                     enc_blocks=tuple(m.get("enc_blocks", [2, 2, 2, 4])),
                     middle_blocks=m.get("middle_blocks", 3),
                     dec_blocks=tuple(m.get("dec_blocks", [1, 1, 1, 2])),
                     time_dim=m.get("time_dim", 64))
    return ResDiffSampler(net, timesteps=m.get("diffusion_levels", 4), steps=m.get("steps", 1),
                          kappa=m.get("kappa", 0.1), eta_power=m.get("eta_power", 2.0))
