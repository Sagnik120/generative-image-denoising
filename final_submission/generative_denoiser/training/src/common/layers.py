"""
Reusable building blocks shared across architectures.

Keeping these in one place means every architecture folder
(a01_nafnet_unet, a02_restormer_lite, a03_pix2pix_gan, a04_tiny_ddpm_sr3, and any
future architecture you add) can import the same well-tested pieces
instead of re-implementing them.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm that operates on (B, C, H, W) tensors by normalizing over C."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        # Statistics are computed in float32: under mixed precision the squared
        # activations inside var() overflow float16 and turn the run into NaNs.
        dtype = x.dtype
        x = x.float()
        mu = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        x = x * self.weight[None, :, None, None].float() + self.bias[None, :, None, None].float()
        return x.to(dtype)


class SimpleGate(nn.Module):
    """NAFNet's activation-free gate: split channels in half, multiply them."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Cheap channel-attention: global average pool -> 1x1 conv -> scale."""

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        w = self.conv(self.pool(x))
        return x * w


class NAFBlock(nn.Module):
    """
    One NAFNet block: DWConv -> SimpleGate -> SCA -> 1x1 conv,
    plus a second sub-block with SimpleGate-based feed-forward.
    Reference: Chen et al., "Simple Baselines for Image Restoration", ECCV 2022.
    """

    def __init__(self, channels: int, ffn_expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        dw_channels = channels * 2
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(dw_channels, dw_channels, kernel_size=3, padding=1,
                                 groups=dw_channels)
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
        self.drop_path = drop_path

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.sg1(y)
        y = self.sca(y)
        y = self.conv2(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv3(y)
        y = self.sg2(y)
        y = self.conv4(y)
        x = x + y * self.gamma
        return x


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.op(x)


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer-style sinusoidal embedding used by DDPM's time conditioning."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device).float() / half
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLM(nn.Module):
    """Feature-wise linear modulation: inject a conditioning vector (e.g. diffusion
    timestep embedding) into a conv feature map via per-channel scale + shift."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, cond):
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return x * (1 + scale) + shift


# --------------------------------------------------------------------------- #
# Blocks added for the round-2 architectures (a05 onward).
# --------------------------------------------------------------------------- #

class ChannelAttention(nn.Module):
    """Restormer's MDTA: attention across CHANNELS, so cost is linear in pixels."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3,
                                     padding=1, groups=channels * 3)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        head_dim = c // self.num_heads
        # The attention products run in float32 even under mixed precision
        # (autocast would otherwise cast the matmuls back to float16).
        with torch.autocast(device_type="cuda", enabled=False):
            q = F.normalize(q.reshape(b, self.num_heads, head_dim, h * w).float(), dim=-1)
            k = F.normalize(k.reshape(b, self.num_heads, head_dim, h * w).float(), dim=-1)
            v = v.reshape(b, self.num_heads, head_dim, h * w).float()
            attn = (q @ k.transpose(-2, -1)) * self.temperature.float()
            out = attn.softmax(dim=-1) @ v
        return self.project_out(out.reshape(b, c, h, w).to(x.dtype))


class SpatialAttention(nn.Module):
    """Global multi-head self-attention over all pixels. Only affordable on the
    coarsest feature maps (16x16 = 256 tokens for a 256x256 input), where it gives
    every position a view of the whole image for a few tens of MFLOPs. Written
    with explicit matmuls so FLOPs counters see the true cost."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        with torch.autocast(device_type="cuda", enabled=False):   # float32, as above
            q, k, v = qkv.float().unbind(1)
            attn = (q.transpose(-2, -1) @ k) * self.scale         # (b, heads, hw, hw)
            out = v @ attn.softmax(dim=-1).transpose(-2, -1)      # (b, heads, d, hw)
        return self.project_out(out.reshape(b, c, h, w).to(x.dtype))


class GatedDconvFFN(nn.Module):
    """Restormer's GDFN: a gated feed-forward with a depthwise conv."""

    def __init__(self, channels: int, expand: float = 2.0):
        super().__init__()
        hidden = int(channels * expand)
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1,
                                 groups=hidden * 2)
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block; `attn` picks channel or spatial attention."""

    def __init__(self, channels: int, num_heads: int = 4, ffn_expand: float = 2.0,
                 attn: str = "channel"):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        attn_cls = SpatialAttention if attn == "spatial" else ChannelAttention
        self.attn = attn_cls(channels, num_heads)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = GatedDconvFFN(channels, ffn_expand)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class FrequencyBranch(nn.Module):
    """Filters the feature map in the Fourier domain. One FFT gives every output
    pixel a global receptive field, which is what undoing blur (a convolution,
    i.e. a per-frequency multiplication) and periodic/JPEG-block patterns needs."""

    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.reduce = nn.Conv2d(channels, mid, kernel_size=1)
        self.conv1 = nn.Conv2d(mid * 2, mid * 2, kernel_size=1)
        self.conv2 = nn.Conv2d(mid * 2, mid * 2, kernel_size=1)
        self.expand = nn.Conv2d(mid, channels, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[-2:]
        y = self.reduce(x)
        dtype = y.dtype
        # FFT and the spectrum convs run in float32 (cuFFT half precision only
        # supports power-of-two sizes and the spectrum has a huge dynamic range).
        with torch.autocast(device_type="cuda", enabled=False):
            spec = torch.fft.rfft2(y.float(), norm="ortho")
            spec = torch.cat([spec.real, spec.imag], dim=1)
            spec = spec + self.conv2(F.gelu(self.conv1(spec)))
            real, imag = spec.chunk(2, dim=1)
            y = torch.fft.irfft2(torch.complex(real, imag), s=(h, w), norm="ortho")
        return self.expand(y.to(dtype))


class DualDomainBlock(nn.Module):
    """A NAFBlock whose second half runs a spatial feed-forward and a
    FrequencyBranch side by side on the same normalised features."""

    def __init__(self, channels: int, ffn_expand: int = 2, freq_reduction: int = 2):
        super().__init__()
        dw_channels = channels * 2
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(dw_channels, dw_channels, kernel_size=3, padding=1,
                                 groups=dw_channels)
        self.sg = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_channels // 2)
        self.conv2 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1)

        ffn_channels = channels * ffn_expand
        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1)
        self.freq = FrequencyBranch(channels, freq_reduction)

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.delta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.conv2(self.sca(self.sg(self.dwconv(self.conv1(self.norm1(x))))))
        x = x + y * self.beta
        y = self.norm2(x)
        return x + self.conv4(self.sg(self.conv3(y))) * self.gamma + self.freq(y) * self.delta


def haar_dwt(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal 2D Haar transform: (B, C, H, W) -> (B, 4C, H/2, W/2), lossless."""
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return torch.cat([(a + b + c + d), (a - b + c - d),
                      (a + b - c - d), (a - b - c + d)], dim=1) * 0.5


def haar_iwt(y: torch.Tensor) -> torch.Tensor:
    """Exact inverse of `haar_dwt`."""
    ll, lh, hl, hh = y.chunk(4, dim=1)
    b, ch, h, w = ll.shape
    out = y.new_zeros(b, ch, h * 2, w * 2)
    out[..., 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
    out[..., 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
    out[..., 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
    out[..., 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
    return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
