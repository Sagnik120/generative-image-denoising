"""
Generative denoiser: conditional GAN.
Generator: NAF U-Net with gated noise injection after every decoder block
(src/common/gennet.py). Discriminator (training only): conditional U-Net with
spectral normalisation on Haar wavelet detail bands (src/common/gan.py).
Losses, diversity constraint and adversarial training: src/common/gan.py.
"""
from src.common.gennet import GenNet, build_generator as build_model
from src.common.gan import GANBundle

NAME = "generative_denoiser"


def build(cfg, device):
    return GANBundle(cfg, device, build_model(cfg.get("model")), name=NAME)
