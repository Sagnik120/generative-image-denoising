"""
a26_resdiff_gan -- one-step residual diffusion + GAN (Denoising Diffusion GANs,
Xiao et al. 2022): a22's network and forward process, trained with the
round-5 stochastic recipe and a conditional RGB discriminator on the
clean-image estimate. 1 sampling step inside the model call.

Round 5 (strictly generative, see docs/PLAN.md). Code: src/common/resdiff.py
(ResDiffGANBundle) and src/common/gan.py.
"""
from src.common.resdiff import ResDiffSampler, ResDiffGANBundle, build_sampler as build_model

NAME = "a26_resdiff_gan"


def build(cfg, device):
    return ResDiffGANBundle(cfg, device, build_model(cfg.get("model")), name=NAME)
