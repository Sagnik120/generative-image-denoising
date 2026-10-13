"""
a20_gan -- conditional GAN: NAF U-Net generator with noise injection, RGB U-Net discriminator.

Round 4 (strictly generative, see docs/PLAN.md). Generator and training are
in src/common/gennet.py and src/common/gan.py; only config.yaml differs
between the GAN models. The discriminator is used in training only.
"""
from src.common.gennet import GenNet, build_generator as build_model
from src.common.gan import GANBundle

NAME = "a20_gan"


def build(cfg, device):
    return GANBundle(cfg, device, build_model(cfg.get("model")), name=NAME)
