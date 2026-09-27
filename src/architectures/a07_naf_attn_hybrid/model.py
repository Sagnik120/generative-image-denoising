"""
a07_naf_attn_hybrid -- convolution where pixels are many, attention where
they are few.

  256 / 128 / 64 px : NAFBlocks. Noise and fine detail are local problems,
                      and attention at these sizes is the expensive part of
                      a06.
  32 px             : channel-attention transformer blocks (Restormer MDTA).
  16 px bottleneck  : GLOBAL spatial self-attention. 16x16 is only 256
                      tokens, so every position can attend to the whole
                      image for a few tens of MFLOPs per block -- the
                      long-range context that large blur kernels and
                      repeated structure need.

The competition fixes the input at 256x256 for both training and testing, so
a global operator like this has no train/test size mismatch -- the usual
reason restoration networks avoid them.
"""
import torch.nn as nn

from src.common.layers import NAFBlock, TransformerBlock
from src.common.unet import RestorationUNet
from src.common.bundle import RestorationBundle

NAME = "a07_naf_attn_hybrid"


def hybrid_block_fn(n_levels, channel_heads=4, spatial_heads=8, ffn_expand=2.0):
    """NAF on levels 0..n-2, channel attention on the last encoder level,
    spatial attention in the bottleneck (level n)."""
    def block(level, ch):
        if level == n_levels:
            return TransformerBlock(ch, spatial_heads, ffn_expand, attn="spatial")
        if level == n_levels - 1:
            return TransformerBlock(ch, channel_heads, ffn_expand, attn="channel")
        return NAFBlock(ch)
    return block


class NAFAttnHybrid(nn.Module):
    def __init__(self, widths=(24, 48, 96, 192, 256), enc_blocks=(2, 2, 2, 2),
                 middle_blocks=3, dec_blocks=(1, 1, 1, 2), predict_residual=True):
        super().__init__()
        self.net = RestorationUNet(hybrid_block_fn(len(enc_blocks)), widths=widths,
                                   enc_blocks=enc_blocks, middle_blocks=middle_blocks,
                                   dec_blocks=dec_blocks, predict_residual=predict_residual)

    def forward(self, x):
        return self.net(x)


def build_model(model_cfg):
    m = dict(model_cfg or {})
    return NAFAttnHybrid(widths=tuple(m.get("widths", [24, 48, 96, 192, 256])),
                         enc_blocks=tuple(m.get("enc_blocks", [2, 2, 2, 2])),
                         middle_blocks=m.get("middle_blocks", 3),
                         dec_blocks=tuple(m.get("dec_blocks", [1, 1, 1, 2])),
                         predict_residual=m.get("predict_residual", True))


def build(cfg, device):
    return RestorationBundle(cfg, device, build_model(cfg.get("model")), name=NAME)
