"""
Generative building blocks for round 4 (a20 onward).

Round 4 replaces regression with strictly generative models: every network
has a random input, and that randomness must change the output. Two pieces
are shared here:

  NoiseInjection - adds per-pixel Gaussian noise to a feature map with a
                   learned per-channel strength (StyleGAN, Karras et al. 2019).
                   The strength starts at zero, so training starts stable and
                   the network learns how much randomness to use.
  GenNet         - the a05/a15 NAF U-Net with a NoiseInjection after every
                   block; the generator of the GAN models.

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
    def __init__(self, channels, res, index=0):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.index = index
        self.register_buffer("eval_noise", torch.zeros(1, 1, res, res))
        self.reseed(0)

    def reseed(self, seed):
        g = torch.Generator().manual_seed(int(seed) * 100003 + self.index)
        noise = torch.randn(self.eval_noise.shape, generator=g)
        self.eval_noise.copy_(noise.to(self.eval_noise.device, self.eval_noise.dtype))

    def forward(self, x):
        b, _, h, w = x.shape
        if self.training or self.eval_noise.shape[-2:] != (h, w):
            noise = torch.randn(b, 1, h, w, device=x.device, dtype=x.dtype)
        else:
            noise = self.eval_noise.to(x.dtype)
        return x + self.weight * noise


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
                 dec_blocks=(1, 1, 1, 2), noise=True, image_size=256):
        super().__init__()
        widths = [width * 2 ** i for i in range(len(enc_blocks) + 1)]
        counter = [0]

        def block(level, ch):
            if not noise:
                return NAFBlock(ch)
            counter[0] += 1
            return nn.Sequential(NAFBlock(ch), NoiseInjection(ch, image_size >> level, counter[0]))

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
                  noise=m.get("noise", True))
