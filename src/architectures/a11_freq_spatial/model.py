"""
a11_freq_spatial -- every block below full resolution works in the spatial
AND the Fourier domain.

Blur is a convolution, which in the Fourier domain is a per-frequency
multiplication; undoing it with 3x3 convolutions needs many stacked layers
to cover the kernel, while one FFT gives each block a global view. The same
holds for other corruptions that are compact in frequency: JPEG's 8x8 block
grid, stripes and periodic patterns. Noise, by contrast, is best handled
locally. So each DualDomainBlock runs both: a NAF spatial path, plus a
branch that transforms the features with an FFT, filters the spectrum with
1x1 convs, and transforms back (the idea behind DeepRFT, Mao et al. 2023,
and FFTformer, Kong et al. 2023).

Full resolution keeps plain NAFBlocks (the FFT branch is most expensive
there). As with a07's global attention, the fixed 256x256 input removes the
train/test size mismatch that normally argues against global operators.

Aimed at the blur and blur+noise+JPEG cases of the benchmark; compare with
a05, which is the same network without the frequency branch.
"""
import torch.nn as nn

from src.common.layers import NAFBlock, DualDomainBlock
from src.common.unet import RestorationUNet
from src.common.bundle import RestorationBundle

NAME = "a11_freq_spatial"


class FreqSpatialNet(nn.Module):
    def __init__(self, width=24, enc_blocks=(2, 2, 2, 2), middle_blocks=2,
                 dec_blocks=(1, 1, 1, 1), freq_reduction=2, predict_residual=True):
        super().__init__()
        widths = [width * 2 ** i for i in range(len(enc_blocks) + 1)]
        block = lambda level, ch: NAFBlock(ch) if level == 0 else DualDomainBlock(
            ch, freq_reduction=freq_reduction)
        self.net = RestorationUNet(block, widths=widths, enc_blocks=enc_blocks,
                                   middle_blocks=middle_blocks, dec_blocks=dec_blocks,
                                   predict_residual=predict_residual)

    def forward(self, x):
        return self.net(x)


def build_model(model_cfg):
    m = dict(model_cfg or {})
    return FreqSpatialNet(width=m.get("width", 24),
                          enc_blocks=tuple(m.get("enc_blocks", [2, 2, 2, 2])),
                          middle_blocks=m.get("middle_blocks", 2),
                          dec_blocks=tuple(m.get("dec_blocks", [1, 1, 1, 1])),
                          freq_reduction=m.get("freq_reduction", 2),
                          predict_residual=m.get("predict_residual", True))


def build(cfg, device):
    return RestorationBundle(cfg, device, build_model(cfg.get("model")), name=NAME)
