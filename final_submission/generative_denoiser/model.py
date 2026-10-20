"""
Generator network of the generative denoiser.

A U-Net built from NAFNet blocks (Chen et al., "Simple Baselines for Image
Restoration", ECCV 2022) that predicts a correction added to the corrupted
input. It is a conditional generator: after each decoder block, Gaussian noise
is injected, multiplied by a learned per-channel strength and by a learned
per-pixel gate (sigmoid of a 1x1 convolution of the block's features). Each
call therefore draws a sample from the learned distribution of clean images
given the corrupted one. The encoder and bottleneck are noise-free.

Input and output: float tensors (B, 3, 256, 256) in [0, 1].
"""
import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of (B, C, H, W) tensors."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        mu = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        x = x * self.weight[None, :, None, None].float() + self.bias[None, :, None, None].float()
        return x.to(dtype)


class SimpleGate(nn.Module):
    """Splits the channels in half and multiplies the halves."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    def __init__(self, channels, ffn_expand=2):
        super().__init__()
        dw_channels = channels * 2
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(dw_channels, dw_channels, kernel_size=3, padding=1, groups=dw_channels)
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_channels // 2)
        self.conv2 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1)
        ffn_channels = channels * ffn_expand
        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn_channels, kernel_size=1)
        self.sg2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.conv2(self.sca(self.sg1(self.dwconv(self.conv1(self.norm1(x))))))
        x = x + y * self.beta
        y = self.conv4(self.sg2(self.conv3(self.norm2(x))))
        return x + y * self.gamma


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.Sequential(nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, bias=False),
                                nn.PixelShuffle(2))

    def forward(self, x):
        return self.op(x)


class NoiseInjection(nn.Module):
    """Adds gated Gaussian noise: x + (strength * scale) * sigmoid(gate(x)) * n.

    fresh=True (default at inference): new noise on every call.
    fresh=False: a fixed noise draw stored in `eval_noise` (set by reseed(seed)),
    so a given seed always gives the same sample.
    `scale` is the test-time noise multiplier.
    """

    def __init__(self, channels, res, index=0, gate=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gate = nn.Conv2d(channels, 1, kernel_size=1) if gate else None
        self.scale = 1.0
        self.fresh = True
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


class UNet(nn.Module):
    def __init__(self, block_fn, widths, enc_blocks, middle_blocks, dec_blocks):
        super().__init__()
        n_levels = len(widths) - 1
        self.intro = nn.Conv2d(3, widths[0], kernel_size=3, padding=1)
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        for lvl, n in enumerate(enc_blocks):
            self.encoders.append(nn.Sequential(*[block_fn(lvl, widths[lvl]) for _ in range(n)]))
            self.downs.append(Downsample(widths[lvl], widths[lvl + 1]))
        self.middle = nn.Sequential(*[block_fn(n_levels, widths[-1]) for _ in range(middle_blocks)])
        self.ups, self.decoders = nn.ModuleList(), nn.ModuleList()
        for i, n in enumerate(dec_blocks):
            lvl = n_levels - 1 - i
            self.ups.append(Upsample(widths[lvl + 1], widths[lvl]))
            self.decoders.append(nn.Sequential(*[block_fn(lvl, widths[lvl]) for _ in range(n)]))
        self.outro = nn.Conv2d(widths[0], 3, kernel_size=3, padding=1)

    def forward(self, x):
        inp, x, skips = x, self.intro(x), []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = dec(up(x) + skip)
        return self.outro(x) + inp          # residual: predicts a correction to the input


class Generator(nn.Module):
    """NAF U-Net generator with gated noise injection after every decoder block."""

    def __init__(self, width=24, enc_blocks=(2, 2, 2, 4), middle_blocks=3, dec_blocks=(1, 1, 1, 2),
                 image_size=256, noise_gate=True):
        super().__init__()
        widths = [width * 2 ** i for i in range(len(enc_blocks) + 1)]
        first_decoder = sum(enc_blocks) + middle_blocks + 1
        counter = [0]

        def block(level, ch):
            counter[0] += 1
            if counter[0] < first_decoder:                       # encoder and bottleneck: no noise
                return NAFBlock(ch)
            return nn.Sequential(NAFBlock(ch), NoiseInjection(ch, image_size >> level, counter[0], noise_gate))

        self.net = UNet(block, widths, tuple(enc_blocks), middle_blocks, tuple(dec_blocks))

    def forward(self, x):
        return self.net(x)


def set_noise_scale(model, scale):
    for m in model.modules():
        if isinstance(m, NoiseInjection):
            m.scale = float(scale)


def set_noise_seed(model, seed):
    """seed=None: fresh noise on every call. int: fixed, reproducible noise."""
    for m in model.modules():
        if isinstance(m, NoiseInjection):
            m.fresh = seed is None
            if seed is not None:
                m.reseed(seed)
