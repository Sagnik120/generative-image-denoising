"""
a24_gan_div -- a20's conditional GAN (RGB U-Net discriminator) + the round-5
stochastic recipe: loss on the mean of two samples, diversity hinge,
adversarial weight 0.01, noise strength starting at 0.01.

Round 5 (strictly generative, see docs/PLAN.md). Code: src/common/gennet.py,
src/common/gan.py; only config.yaml differs between the GAN models.
"""
from src.common.gennet import GenNet, build_generator as build_model
from src.common.gan import GANBundle

NAME = "a24_gan_div"


def build(cfg, device):
    return GANBundle(cfg, device, build_model(cfg.get("model")), name=NAME)
