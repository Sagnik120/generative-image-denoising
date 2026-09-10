"""
NAFNet-style U-Net -- the recommended primary architecture.

Why this one first (see docs/nafnet_unet/notes.md for the full rationale
pulled from the literature review):
  - NAFNet (Chen et al., ECCV 2022) strips out expensive nonlinear
    activations, replacing them with a cheap "Simple Gate" -- this measures
    substantially lower FLOPs than Transformer backbones like Restormer at
    matched quality (literature review Section 5.2).
  - It is still a plain U-Net-shaped encoder-decoder with skip connections
    (the backbone reused by nearly every architecture in the review,
    Section 6), so it is simple, well understood, and fast to iterate on.
  - Predicting the noise/residual instead of the clean image directly
    (DDPM-style objective, configurable below) tends to train more stably
    for denoising tasks.

This is a *conditional generator*: it is trained as a single deterministic
forward pass, corrupted image -> clean image, which is the "one denoising
step of a diffusion model" framing recommended in the literature review's
Section 7 -- giving strong quality without paying multi-step diffusion
sampling cost (keeps FLOPs low, matching the competition's efficiency axis).
"""
import torch
import torch.nn as nn

from src.common.layers import NAFBlock, Downsample, Upsample, count_parameters
from src.common.losses import build_reconstruction_loss


class NAFNetUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, width=32, enc_blocks=(2, 2, 4, 8),
                 middle_blocks=4, dec_blocks=(2, 2, 2, 2), predict_residual=True):
        super().__init__()
        self.predict_residual = predict_residual

        self.intro = nn.Conv2d(in_ch, width, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for n_blocks in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n_blocks)]))
            self.downs.append(Downsample(ch, ch * 2))
            ch *= 2

        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blocks)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n_blocks in dec_blocks:
            self.ups.append(Upsample(ch, ch // 2))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n_blocks)]))

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


class NAFNetBundle:
    """ModelBundle wrapper implementing the Trainer interface (see common/trainer.py)."""

    def __init__(self, cfg, device):
        m = cfg["model"]
        self.device = device
        self.model = NAFNetUNet(
            width=m.get("width", 32),
            enc_blocks=tuple(m.get("enc_blocks", [2, 2, 4, 8])),
            middle_blocks=m.get("middle_blocks", 4),
            dec_blocks=tuple(m.get("dec_blocks", [2, 2, 2, 2])),
            predict_residual=m.get("predict_residual", True),
        ).to(device)

        self.criterion = build_reconstruction_loss(cfg["train"].get("loss", "charbonnier"))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["train"].get("lr", 2e-4),
            betas=(0.9, 0.9),
            weight_decay=cfg["train"].get("weight_decay", 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg["train"]["epochs"],
            eta_min=cfg["train"].get("lr_min", 1e-6),
        )
        print(f"[nafnet_unet] Model parameters: {count_parameters(self.model):,}")

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
    return NAFNetBundle(cfg, device)
