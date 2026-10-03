"""
a05_nafnet_v2 -- the pure-convolution reference for round 2.

Same NAFBlock as a01 (Chen et al., "Simple Baselines for Image Restoration",
ECCV 2022), but re-proportioned for a ~5 GFLOPs budget instead of a01's 12.8:
narrower (width 24), one level deeper (down to 16x16), and with most blocks
moved into the encoder's coarse levels, where each block sees a larger part
of the image for the same cost.

Its job in the comparison is to answer "how far does a plain CNN get under
the new data, degradations and losses?" -- every other round-2 architecture
is measured against it at the same FLOPs.
"""
import torch.nn as nn

from src.common.layers import NAFBlock
from src.common.unet import RestorationUNet
from src.common.bundle import RestorationBundle

NAME = "a05_nafnet_v2"


class NAFNetV2(nn.Module):
    def __init__(self, width=24, enc_blocks=(2, 2, 2, 4), middle_blocks=3,
                 dec_blocks=(1, 1, 1, 2), predict_residual=True):
        super().__init__()
        widths = [width * 2 ** i for i in range(len(enc_blocks) + 1)]
        self.net = RestorationUNet(lambda level, ch: NAFBlock(ch), widths=widths,
                                   enc_blocks=enc_blocks, middle_blocks=middle_blocks,
                                   dec_blocks=dec_blocks, predict_residual=predict_residual)

    def forward(self, x):
        return self.net(x)


def build_model(model_cfg):
    m = dict(model_cfg or {})
    return NAFNetV2(width=m.get("width", 24),
                    enc_blocks=tuple(m.get("enc_blocks", [2, 2, 2, 4])),
                    middle_blocks=m.get("middle_blocks", 3),
                    dec_blocks=tuple(m.get("dec_blocks", [1, 1, 1, 2])),
                    predict_residual=m.get("predict_residual", True))


def build(cfg, device):
    return RestorationBundle(cfg, device, build_model(cfg.get("model")), name=NAME)
