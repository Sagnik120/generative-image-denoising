"""
Tiny conditional diffusion model (SR3-style, few-step).

Based on Ho et al. (DDPM, NeurIPS 2020) and Saharia et al. (SR3, 2022),
reviewed in the literature review Sections 4.1 and 4.3. Two ideas are
combined:
  1. DDPM's training objective: a U-Net learns to predict the noise added
     at a given timestep.
  2. SR3's conditioning trick: the corrupted image is concatenated as extra
     input channels at every step, so the network always knows what it is
     trying to restore (not just denoise arbitrary Gaussian noise).

IMPORTANT DESIGN CHOICE FOR THIS COMPETITION: a full DDPM uses ~1000
sampling steps, which would be far too expensive for the FLOPs axis (each
step is a full forward pass). Per the literature review's Section 7
recommendation, this implementation uses a SMALL number of diffusion
steps (default T=8, configurable) -- enough to get some benefit from the
iterative-refinement idea and the generative framing, while keeping total
inference FLOPs to (T x one U-Net forward pass), which stays competitive
if the per-step U-Net is kept small.

This is the most experimental of the four architectures here: expect it to
be more finicky to train than nafnet_unet or restormer_lite, and to cost
more FLOPs at inference (T forward passes instead of 1) for a potential
perceptual-quality benefit. Compare all four in results/ before deciding
which one to submit.
"""
import math
import torch
import torch.nn as nn

from src.common.layers import (NAFBlock, Downsample, Upsample, FiLM,
                                sinusoidal_time_embedding, count_parameters)
from src.common.losses import build_reconstruction_loss


class TimeConditionedNAFBlock(nn.Module):
    """A NAFBlock followed by FiLM conditioning on the diffusion timestep embedding."""

    def __init__(self, channels, time_emb_dim):
        super().__init__()
        self.block = NAFBlock(channels)
        self.film = FiLM(time_emb_dim, channels)

    def forward(self, x, t_emb):
        x = self.block(x)
        return self.film(x, t_emb)


class ConditionalUNet(nn.Module):
    """U-Net that predicts noise, conditioned on both the diffusion timestep and the
    corrupted image (concatenated as extra input channels, SR3-style)."""

    def __init__(self, img_ch=3, cond_ch=3, width=24, enc_blocks=(2, 2, 3),
                 middle_blocks=3, dec_blocks=(3, 2, 2), time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        in_ch = img_ch + cond_ch  # noisy image + corrupted-image condition
        self.intro = nn.Conv2d(in_ch, width, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for n_blocks in enc_blocks:
            self.encoders.append(nn.ModuleList(
                [TimeConditionedNAFBlock(ch, time_emb_dim) for _ in range(n_blocks)]))
            self.downs.append(Downsample(ch, ch * 2))
            ch *= 2

        self.middle = nn.ModuleList(
            [TimeConditionedNAFBlock(ch, time_emb_dim) for _ in range(middle_blocks)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n_blocks in dec_blocks:
            self.ups.append(Upsample(ch, ch // 2))
            ch //= 2
            self.decoders.append(nn.ModuleList(
                [TimeConditionedNAFBlock(ch, time_emb_dim) for _ in range(n_blocks)]))

        self.outro = nn.Conv2d(width, img_ch, kernel_size=3, padding=1)

    def forward(self, noisy_img, cond_img, t):
        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        x = torch.cat([noisy_img, cond_img], dim=1)
        x = self.intro(x)

        skips = []
        for stage, down in zip(self.encoders, self.downs):
            for blk in stage:
                x = blk(x, t_emb)
            skips.append(x)
            x = down(x)

        for blk in self.middle:
            x = blk(x, t_emb)

        for up, stage, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = x + skip
            for blk in stage:
                x = blk(x, t_emb)

        return self.outro(x)


def make_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T)


class TinyDDPM(nn.Module):
    """Wraps ConditionalUNet with the DDPM forward/reverse process, restricted to a
    small number of timesteps T so inference stays cheap (T forward passes total)."""

    def __init__(self, T=8, width=24, enc_blocks=(2, 2, 3), middle_blocks=3,
                 dec_blocks=(3, 2, 2), time_emb_dim=128):
        super().__init__()
        self.T = T
        self.unet = ConditionalUNet(width=width, enc_blocks=enc_blocks,
                                     middle_blocks=middle_blocks, dec_blocks=dec_blocks,
                                     time_emb_dim=time_emb_dim)

        betas = make_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def q_sample(self, x0, t, noise):
        """Forward diffusion: add noise to x0 according to timestep t."""
        sqrt_acp = self.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_acp = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_acp * x0 + sqrt_one_minus_acp * noise

    def predict_noise(self, x_t, cond_img, t):
        return self.unet(x_t, cond_img, t)

    @torch.no_grad()
    def sample(self, cond_img):
        """Reverse process: start from pure noise, run T denoising steps conditioned
        on the corrupted image, return the final estimate of the clean image."""
        b = cond_img.shape[0]
        x = torch.randn_like(cond_img)
        for step in reversed(range(self.T)):
            t = torch.full((b,), step, device=cond_img.device, dtype=torch.long)
            beta_t = self.betas[step]
            alpha_t = self.alphas[step]
            alpha_cumprod_t = self.alphas_cumprod[step]

            pred_noise = self.predict_noise(x, cond_img, t)
            coef1 = 1.0 / alpha_t.sqrt()
            coef2 = beta_t / (1 - alpha_cumprod_t).sqrt()
            mean = coef1 * (x - coef2 * pred_noise)

            if step > 0:
                noise = torch.randn_like(x)
                x = mean + beta_t.sqrt() * noise
            else:
                x = mean
        return x


class SingleStepFlopsWrapper(nn.Module):
    """
    The common FLOPs counter (src/common/metrics.py) calls model(x) with a single
    (1, 3, 256, 256) tensor, matching the competition's measurement protocol. This
    diffusion U-Net's real forward signature needs three inputs (noisy image,
    condition image, timestep) -- this wrapper adapts it to a single-tensor call by
    reusing the same tensor as both the noisy and condition image at a fixed
    mid-range timestep, so FLOPs-per-step can still be measured the same way as the
    other architectures. The TRUE inference cost is `T x this value` (T diffusion
    steps) -- this is reported explicitly alongside the raw number in
    metrics/final_report.json, so it is never silently underreported.
    """

    def __init__(self, unet, T):
        super().__init__()
        self.unet = unet
        self.T = T

    def forward(self, x):
        b = x.shape[0]
        t = torch.full((b,), self.T // 2, device=x.device, dtype=torch.long)
        return self.unet(x, x, t)


class TinyDDPMBundle:
    def __init__(self, cfg, device):
        m = cfg["model"]
        self.device = device
        self.T = m.get("timesteps", 8)
        self.model = TinyDDPM(
            T=self.T,
            width=m.get("width", 24),
            enc_blocks=tuple(m.get("enc_blocks", [2, 2, 3])),
            middle_blocks=m.get("middle_blocks", 3),
            dec_blocks=tuple(m.get("dec_blocks", [3, 2, 2])),
            time_emb_dim=m.get("time_emb_dim", 128),
        ).to(device)

        self.criterion = build_reconstruction_loss(cfg["train"].get("loss", "l2"))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["train"].get("lr", 2e-4),
            weight_decay=cfg["train"].get("weight_decay", 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg["train"]["epochs"],
            eta_min=cfg["train"].get("lr_min", 1e-6),
        )
        print(f"[tiny_ddpm_sr3] Model parameters: {count_parameters(self.model):,}, T={self.T}")

    def train_step(self, corrupted, clean):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        b = clean.shape[0]
        t = torch.randint(0, self.T, (b,), device=self.device).long()
        noise = torch.randn_like(clean)
        x_t = self.model.q_sample(clean, t, noise)

        pred_noise = self.model.predict_noise(x_t, corrupted, t)
        loss = self.criterion(pred_noise, noise)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return {"loss": loss.item()}

    @torch.no_grad()
    def eval_step(self, corrupted, clean):
        self.model.eval()
        pred = self.model.sample(corrupted)
        # Report reconstruction loss (not noise-prediction loss) for a metric that's
        # comparable across architectures (they all show pixel-space L1/L2 loss).
        loss = torch.nn.functional.l1_loss(pred.clamp(0, 1), clean)
        return pred.clamp(0, 1), {"loss": loss.item()}

    def get_inference_model(self):
        # See SingleStepFlopsWrapper docstring: reports ONE diffusion step's FLOPs;
        # true inference cost is (T x this value), T = self.T diffusion steps.
        return SingleStepFlopsWrapper(self.model.unet, self.T)

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
    return TinyDDPMBundle(cfg, device)
