"""
a27_full -- a27 trained 200k steps on data v3 (round 7). Otherwise a25's generator and wavelet-domain discriminator with the
round-6 recipe: full composite loss on every output (as a20), 16 images per
step, a second noise draw for 4 of them and an active diversity constraint
(target 1.5 grey levels), adversarial weight 0.005 from step 0.

Round 7 (strictly generative, see docs/PLAN.md). Code: src/common/gennet.py,
src/common/gan.py; only config.yaml differs between the GAN models.
"""
from src.common.gennet import GenNet, build_generator as build_model
from src.common.gan import GANBundle

NAME = "a27_full"


def build(cfg, device):
    return GANBundle(cfg, device, build_model(cfg.get("model")), name=NAME)
