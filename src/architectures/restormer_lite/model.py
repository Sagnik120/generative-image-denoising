"""
Restormer-lite -- an efficient-Transformer U-Net.

Based on Zamir et al., "Restormer: Efficient Transformer for High-Resolution
Image Restoration" (CVPR 2022), reviewed in the literature review Section 5.1.
The key trick reused here is Multi-Dconv Head Transposed Attention (MDTA):
attention is computed across CHANNELS instead of spatial positions, which
gives linear (not quadratic) complexity in image size -- this is what makes
Transformer-quality restoration feasible at 256x256 without exploding FLOPs.

Trade-off vs. nafnet_unet: typically higher quality ceiling (especially on
structure/perceptual metrics like DISTS) at a higher FLOPs cost -- see the
literature review's FLOPs comparison (Restormer ~1128.9 GFLOPs vs NAFNet
~505.5 GFLOPs at 512x512). Train both and compare in results/ to decide
which trade-off wins for your submission.
"""
import torch
import torch.nn as nn

from src.common.layers import LayerNorm2d, Downsample, Upsample, count_parameters
from src.common.losses import build_reconstruction_loss


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention: attention across channels, not pixels."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3,
                                     padding=1, groups=channels * 3)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = c // self.num_heads
        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.reshape(b, c, h, w)
        return self.project_out(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network: a gated MLP with depthwise convs."""

    def __init__(self, channels, expand=2.66):
        super().__init__()
        hidden = int(channels * expand)
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1,
                                 groups=hidden * 2)
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = torch.nn.functional.gelu(x1) * x2
        return self.project_out(x)


class RestormerBlock(nn.Module):
    def __init__(self, channels, num_heads=4, ffn_expand=2.66):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = GDFN(channels, ffn_expand)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RestormerLite(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, width=24, enc_blocks=(2, 3, 3),
                 middle_blocks=4, dec_blocks=(3, 3, 2), num_heads=(1, 2, 4),
                 predict_residual=True):
        super().__init__()
        self.predict_residual = predict_residual
        self.intro = nn.Conv2d(in_ch, width, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for n_blocks, heads in zip(enc_blocks, num_heads):
            self.encoders.append(nn.Sequential(
                *[RestormerBlock(ch, num_heads=heads) for _ in range(n_blocks)]))
            self.downs.append(Downsample(ch, ch * 2))
            ch *= 2

        self.middle = nn.Sequential(
            *[RestormerBlock(ch, num_heads=num_heads[-1] * 2) for _ in range(middle_blocks)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n_blocks, heads in zip(dec_blocks, reversed(num_heads)):
            self.ups.append(Upsample(ch, ch // 2))
            ch //= 2
            self.decoders.append(nn.Sequential(
                *[RestormerBlock(ch, num_heads=heads) for _ in range(n_blocks)]))

        self.outro = nn.Conv2d(width, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        inp = x
        x = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = x + skip
            x = dec(x)

        out = self.outro(x)
        if self.predict_residual:
            out = inp + out
        return out


class RestormerLiteBundle:
    def __init__(self, cfg, device):
        m = cfg["model"]
        self.device = device
        self.model = RestormerLite(
            width=m.get("width", 24),
            enc_blocks=tuple(m.get("enc_blocks", [2, 3, 3])),
            middle_blocks=m.get("middle_blocks", 4),
            dec_blocks=tuple(m.get("dec_blocks", [3, 3, 2])),
            num_heads=tuple(m.get("num_heads", [1, 2, 4])),
            predict_residual=m.get("predict_residual", True),
        ).to(device)

        self.criterion = build_reconstruction_loss(cfg["train"].get("loss", "charbonnier"))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["train"].get("lr", 3e-4),
            betas=(0.9, 0.999),
            weight_decay=cfg["train"].get("weight_decay", 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg["train"]["epochs"],
            eta_min=cfg["train"].get("lr_min", 1e-6),
        )
        print(f"[restormer_lite] Model parameters: {count_parameters(self.model):,}")

    def train_step(self, corrupted, clean):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        pred = self.model(corrupted)
        loss = self.criterion(pred, clean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return {"loss": loss.item()}

    @torch.no_grad()
    def eval_step(self, corrupted, clean):
        self.model.eval()
        pred = self.model(corrupted)
        loss = self.criterion(pred, clean)
        return pred.clamp(0, 1), {"loss": loss.item()}

    def get_inference_model(self):
        return self.model

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def step_scheduler(self):
        self.scheduler.step()

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])


def build(cfg, device):
    return RestormerLiteBundle(cfg, device)
