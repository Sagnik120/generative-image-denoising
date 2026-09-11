"""
Pix2Pix-style conditional GAN.

Based on Isola et al., "Image-to-Image Translation with Conditional
Adversarial Networks" (CVPR 2017), reviewed in the literature review
Section 3.1. A U-Net generator (built from the same NAFBlock backbone used
in nafnet_unet, so it's cheap) is trained with a combination of:
  - Reconstruction loss (L1 / Charbonnier) against the clean image
  - Adversarial loss from a PatchGAN discriminator that looks at
    (corrupted, output) vs (corrupted, real-clean) pairs

Why try this: the literature review notes GAN-based adversarial training
tends to help *perceptual* quality (sharper textures, less "safe blurry
averaging") specifically -- this is likely to help your DISTS score, at
the cost of extra training complexity (NOT extra inference FLOPs, since
the discriminator is thrown away after training and only the generator
ships in your final submission).
"""
import torch
import torch.nn as nn

from src.common.layers import NAFBlock, Downsample, Upsample, count_parameters
from src.common.losses import build_reconstruction_loss, gan_loss


class UNetGenerator(nn.Module):
    """Same lightweight NAFBlock-based U-Net used in nafnet_unet -- reused here as the
    Pix2Pix generator so the two architectures are directly comparable."""

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


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator: classifies overlapping patches as real/fake,
    conditioned on the corrupted input image (concatenated channel-wise)."""

    def __init__(self, in_ch=6, base_ch=64, n_layers=3):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, base_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_ch
        for i in range(1, n_layers):
            next_ch = min(ch * 2, 512)
            layers += [
                nn.Conv2d(ch, next_ch, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(next_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = next_ch
        next_ch = min(ch * 2, 512)
        layers += [
            nn.Conv2d(ch, next_ch, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(next_ch),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        layers += [nn.Conv2d(next_ch, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, corrupted, target_or_output):
        x = torch.cat([corrupted, target_or_output], dim=1)
        return self.model(x)


class Pix2PixGANBundle:
    def __init__(self, cfg, device):
        m = cfg["model"]
        self.device = device
        self.generator = UNetGenerator(
            width=m.get("width", 32),
            enc_blocks=tuple(m.get("enc_blocks", [2, 2, 4, 8])),
            middle_blocks=m.get("middle_blocks", 4),
            dec_blocks=tuple(m.get("dec_blocks", [2, 2, 2, 2])),
            predict_residual=m.get("predict_residual", True),
        ).to(device)
        self.discriminator = PatchDiscriminator(
            in_ch=6, base_ch=m.get("disc_base_ch", 64), n_layers=m.get("disc_layers", 3),
        ).to(device)

        self.recon_criterion = build_reconstruction_loss(cfg["train"].get("loss", "l1"))
        self.lambda_recon = cfg["train"].get("lambda_recon", 100.0)

        self.opt_g = torch.optim.Adam(self.generator.parameters(),
                                       lr=cfg["train"].get("lr_g", 2e-4), betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(),
                                       lr=cfg["train"].get("lr_d", 2e-4), betas=(0.5, 0.999))

        epochs = cfg["train"]["epochs"]
        self.sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_g, T_max=epochs)
        self.sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_d, T_max=epochs)

        print(f"[pix2pix_gan] Generator parameters: {count_parameters(self.generator):,}")
        print(f"[pix2pix_gan] Discriminator parameters: {count_parameters(self.discriminator):,}")

    def train_step(self, corrupted, clean):
        self.generator.train()
        self.discriminator.train()

        # ---- Train Discriminator ----
        self.opt_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake = self.generator(corrupted)
        pred_real = self.discriminator(corrupted, clean)
        pred_fake = self.discriminator(corrupted, fake.detach())
        d_loss = 0.5 * (gan_loss(pred_real, True) + gan_loss(pred_fake, False))
        d_loss.backward()
        self.opt_d.step()

        # ---- Train Generator ----
        self.opt_g.zero_grad(set_to_none=True)
        fake = self.generator(corrupted)
        pred_fake_for_g = self.discriminator(corrupted, fake)
        adv_loss = gan_loss(pred_fake_for_g, True)
        recon_loss = self.recon_criterion(fake, clean)
        g_loss = adv_loss + self.lambda_recon * recon_loss
        g_loss.backward()
        self.opt_g.step()

        return {
            "loss": recon_loss.item(),  # primary curve = reconstruction quality
            "gen_loss": g_loss.item(),
            "disc_loss": d_loss.item(),
        }

    @torch.no_grad()
    def eval_step(self, corrupted, clean):
        self.generator.eval()
        pred = self.generator(corrupted)
        loss = self.recon_criterion(pred, clean)
        return pred.clamp(0, 1), {"loss": loss.item()}

    def get_inference_model(self):
        return self.generator  # only the generator ships -- discriminator is training-only

    def get_lr(self):
        return self.opt_g.param_groups[0]["lr"]

    def step_scheduler(self):
        self.sched_g.step()
        self.sched_d.step()

    def state_dict(self):
        return {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
        }

    def load_state_dict(self, state):
        self.generator.load_state_dict(state["generator"])
        self.discriminator.load_state_dict(state["discriminator"])
        if "opt_g" in state:
            self.opt_g.load_state_dict(state["opt_g"])
        if "opt_d" in state:
            self.opt_d.load_state_dict(state["opt_d"])


def build(cfg, device):
    return Pix2PixGANBundle(cfg, device)
