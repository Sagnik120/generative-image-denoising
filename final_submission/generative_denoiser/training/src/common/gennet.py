"""
Generative building blocks for round 4 (a20 onward).

Round 4 replaces regression with strictly generative models: every network
has a random input, and that randomness must change the output. Two pieces
are shared here:

  NoiseInjection - adds per-pixel Gaussian noise to a feature map with a
                   learned per-channel strength (StyleGAN, Karras et al. 2019).
                   The strength starts at zero (round 4) or at a small value
                   (round 5, `noise_init`), and the network learns how much
                   randomness to use. `scale` multiplies the noise at test
                   time only (noise-scale sweep; not saved with the weights).
  GenNet         - the a05/a15 NAF U-Net with a NoiseInjection after every
                   block (or, round 7, only after the decoder blocks); the
                   generator of the GAN models.

Round 7 (a29): `gate=True` multiplies the noise by a learned per-pixel gate,
sigmoid(1x1 conv of the block's features), so the network can add variation
where the image is uncertain and none where it is not. `noise_where="decoder"`
keeps the encoder and bottleneck noise-free, so the skip connections carry
clean features (StyleGAN puts noise only in the synthesis path).

Seeded noise: while training, noise is drawn fresh each call. In eval mode
each noise source uses a fixed buffer generated from a seed, so inference is
reproducible, the buffers travel with the weights, and FLOPs counters can
trace the model. `reseed_noise(model, seed)` swaps in another draw (used to
measure sample diversity).
"""
import torch
import torch.nn as nn

from .layers import NAFBlock
from .unet import RestorationUNet


class NoiseInjection(nn.Module):
    def __init__(self, channels, res, index=0, init=0.0, gate=False):
        super().__init__()
        self.weight = nn.Parameter(torch.full((1, channels, 1, 1), float(init)))
        self.gate = None
        if gate:                     # per-pixel gate; bias 0 and small weights -> ~0.5 at start
            self.gate = nn.Conv2d(channels, 1, kernel_size=1)
            nn.init.normal_(self.gate.weight, std=1e-3)
            nn.init.zeros_(self.gate.bias)
        self.scale = 1.0
        self.fresh = False           # True: new noise every call in eval mode too (submission)
        self.index = index
        self.register_buffer("eval_noise", torch.zeros(1, 1, res, res))
        self.reseed(0)

    def reseed(self, seed):
        g = torch.Generator().manual_seed(int(seed) * 100003 + self.index)
        noise = torch.randn(self.eval_noise.shape, generator=g)
        self.eval_noise.copy_(noise.to(self.eval_noise.device, self.eval_noise.dtype))

    def forward(self, x):
        b, _, h, w = x.shape
        if self.training or self.fresh or self.eval_noise.shape[-2:] != (h, w):
            noise = torch.randn(b, 1, h, w, device=x.device, dtype=x.dtype)
        else:
            noise = self.eval_noise.to(x.dtype)
        if self.gate is not None:
            noise = noise * torch.sigmoid(self.gate(x))
        return x + (self.weight * self.scale) * noise


def set_noise_scale(model, scale):
    """Multiplies every noise source in `model` by `scale` (evaluation only)."""
    for m in model.modules():
        if hasattr(m, "scale") and hasattr(m, "reseed"):
            m.scale = float(scale)
        if hasattr(m, "noise_scale"):
            m.noise_scale = float(scale)


def set_fresh_noise(model, fresh=True):
    """fresh=True: every call draws new noise (a true sampler, used by the
    submission). fresh=False: the seeded buffers are used (reproducible)."""
    for m in model.modules():
        if hasattr(m, "fresh"):
            m.fresh = bool(fresh)


def reseed_noise(model, seed):
    """Gives every seeded noise source in `model` a new fixed draw."""
    for m in model.modules():
        if m is not model and hasattr(m, "reseed"):
            m.reseed(seed)
    if hasattr(model, "reseed"):
        model.reseed(seed)


class GenNet(nn.Module):
    """NAF U-Net generator: corrupted image + injected noise -> restored image."""

    def __init__(self, width=24, enc_blocks=(2, 2, 2, 4), middle_blocks=3,
                 dec_blocks=(1, 1, 1, 2), noise=True, image_size=256, noise_init=0.0,
                 noise_where="all", noise_gate=False):
        super().__init__()
        widths = [width * 2 ** i for i in range(len(enc_blocks) + 1)]
        counter = [0]
        # RestorationUNet builds encoder, middle, then decoder blocks in order.
        first_decoder = sum(enc_blocks) + middle_blocks + 1

        def block(level, ch):
            counter[0] += 1
            if not noise or (noise_where == "decoder" and counter[0] < first_decoder):
                return NAFBlock(ch)
            return nn.Sequential(NAFBlock(ch), NoiseInjection(ch, image_size >> level, counter[0],
                                                              noise_init, gate=noise_gate))

        self.net = RestorationUNet(block, in_ch=3, out_ch=3, widths=widths,
                                   enc_blocks=enc_blocks, middle_blocks=middle_blocks,
                                   dec_blocks=dec_blocks, predict_residual=True)

    def forward(self, x):
        return self.net(x)


def build_generator(model_cfg):
    m = dict(model_cfg or {})
    return GenNet(width=m.get("width", 24),
                  enc_blocks=tuple(m.get("enc_blocks", [2, 2, 2, 4])),
                  middle_blocks=m.get("middle_blocks", 3),
                  dec_blocks=tuple(m.get("dec_blocks", [1, 1, 1, 2])),
                  noise=m.get("noise", True),
                  noise_init=m.get("noise_init", 0.0),
                  noise_where=m.get("noise_where", "all"),
                  noise_gate=m.get("noise_gate", False))
