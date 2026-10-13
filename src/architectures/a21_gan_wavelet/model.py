"""
a21_gan_wavelet -- a20 with a wavelet-domain discriminator and a wavelet fidelity loss (Korkmaz et al., CVPR 2024).

Round 4 (strictly generative, see docs/PLAN.md). Generator and training are
in src/common/gennet.py and src/common/gan.py; only config.yaml differs
between the GAN models. The discriminator is used in training only.
"""
from src.common.gennet import GenNet, build_generator as build_model
from src.common.gan import GANBundle

NAME = "a21_gan_wavelet"


def build(cfg, device):
    return GANBundle(cfg, device, build_model(cfg.get("model")), name=NAME)
