"""
a09_wavelet_naf -- the same NAF U-Net, run in the Haar wavelet domain.

A Haar transform turns the 256x256x3 image into 128x128x12 with no loss of
information (it is exactly invertible). Every block then runs on a quarter
of the pixels, so the same FLOPs buy roughly a third more blocks than a05.
The network predicts a correction to the wavelet coefficients and the
inverse transform returns to 256x256.

Why wavelets rather than a plain strided conv: the transform already
separates smooth content (LL band) from edges and noise (LH/HL/HH bands),
which is the split a denoiser has to learn anyway, and nothing is thrown
away going down or coming back up.

The question it answers: is full-resolution processing worth its cost, or
is depth at half resolution the better use of the FLOPs budget? If a09
matches a05, the same idea makes a ~2.5 GFLOPs model nearly free.
"""
import torch.nn as nn

from src.common.layers import NAFBlock, haar_dwt, haar_iwt
from src.common.unet import RestorationUNet
from src.common.bundle import RestorationBundle

NAME = "a09_wavelet_naf"


class WaveletNAF(nn.Module):
    def __init__(self, width=40, enc_blocks=(4, 5, 7), middle_blocks=5, dec_blocks=(3, 2, 2)):
        super().__init__()
        widths = [width * 2 ** i for i in range(len(enc_blocks) + 1)]
        self.net = RestorationUNet(lambda level, ch: NAFBlock(ch), in_ch=12, out_ch=12,
                                   widths=widths, enc_blocks=enc_blocks,
                                   middle_blocks=middle_blocks, dec_blocks=dec_blocks,
                                   predict_residual=True)

    def forward(self, x):
        return haar_iwt(self.net(haar_dwt(x)))


def build_model(model_cfg):
    m = dict(model_cfg or {})
    return WaveletNAF(width=m.get("width", 40),
                      enc_blocks=tuple(m.get("enc_blocks", [4, 5, 7])),
                      middle_blocks=m.get("middle_blocks", 5),
                      dec_blocks=tuple(m.get("dec_blocks", [3, 2, 2])))


def build(cfg, device):
    return RestorationBundle(cfg, device, build_model(cfg.get("model")), name=NAME)
