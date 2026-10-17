"""
a29_tuned_30k -- a29_gan_decnoise (round 8, hyperparameter tuning; config.yaml only differs). a27_full with noise injected only after the decoder blocks
and multiplied by a learned per-pixel gate (round 7, Option L). Encoder and
bottleneck are noise-free; data v3, 200k steps, same losses as a27_full.

Round 7 (strictly generative, see docs/PLAN.md). Code: src/common/gennet.py,
src/common/gan.py; only config.yaml differs between the GAN models.
"""
from src.common.gennet import GenNet, build_generator as build_model
from src.common.gan import GANBundle

NAME = "a29_tuned_30k"


def build(cfg, device):
    return GANBundle(cfg, device, build_model(cfg.get("model")), name=NAME)
