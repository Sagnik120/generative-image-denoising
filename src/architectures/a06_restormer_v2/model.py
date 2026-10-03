"""
a06_restormer_v2 -- the pure-transformer reference for round 2.

Restormer blocks (Zamir et al., CVPR 2022): channel-wise attention (MDTA)
plus a gated depthwise-conv feed-forward (GDFN), at every resolution. A
transformer block costs about twice a NAFBlock at the same width, so at
~5 GFLOPs this network is narrower (width 20) and shallower than a05.

Its job in the comparison is the mirror of a05: "does attention everywhere
beat convolution everywhere at equal FLOPs on mixed degradations?"
"""
import torch.nn as nn

from src.common.layers import TransformerBlock
from src.common.unet import RestorationUNet
from src.common.bundle import RestorationBundle

NAME = "a06_restormer_v2"


class RestormerV2(nn.Module):
    def __init__(self, width=20, enc_blocks=(2, 2, 3), middle_blocks=3, dec_blocks=(1, 1, 2),
                 num_heads=(1, 2, 4, 8), ffn_expand=2.0, predict_residual=True):
        super().__init__()
        widths = [width * 2 ** i for i in range(len(enc_blocks) + 1)]
        block = lambda level, ch: TransformerBlock(ch, num_heads[level], ffn_expand, attn="channel")
        self.net = RestorationUNet(block, widths=widths, enc_blocks=enc_blocks,
                                   middle_blocks=middle_blocks, dec_blocks=dec_blocks,
                                   predict_residual=predict_residual)

    def forward(self, x):
        return self.net(x)


def build_model(model_cfg):
    m = dict(model_cfg or {})
    return RestormerV2(width=m.get("width", 20),
                       enc_blocks=tuple(m.get("enc_blocks", [2, 2, 3])),
                       middle_blocks=m.get("middle_blocks", 3),
                       dec_blocks=tuple(m.get("dec_blocks", [1, 1, 2])),
                       num_heads=tuple(m.get("num_heads", [1, 2, 4, 8])),
                       ffn_expand=m.get("ffn_expand", 2.0),
                       predict_residual=m.get("predict_residual", True))


def build(cfg, device):
    return RestorationBundle(cfg, device, build_model(cfg.get("model")), name=NAME)
