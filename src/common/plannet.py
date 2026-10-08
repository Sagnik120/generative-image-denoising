"""
PlanNet -- the one network behind every round-3 architecture (a12 onward).

Round 2 showed that at equal FLOPs the plain NAF U-Net (a05) is as good as
anything else, that a global-attention bottleneck (a07) helps on corruptions
spanning the whole image, and that running in the Haar domain (a09) costs
nothing. PlanNet is a05 with those two ideas as switches, so the round-3
experiments and the final model differ only in their config:

    width / enc_blocks / middle_blocks / dec_blocks : the a05 U-Net
    global_attn : bottleneck (16x16) blocks become global spatial attention;
                  the bottleneck is narrowed to 2/3 to keep FLOPs level
    wavelet     : Haar transform in front (256x256x3 -> 128x128x12), as a09
"""
import torch.nn as nn

from .layers import NAFBlock, TransformerBlock, haar_dwt, haar_iwt
from .unet import RestorationUNet


class PlanNet(nn.Module):
    def __init__(self, width=24, enc_blocks=(2, 2, 2, 4), middle_blocks=3,
                 dec_blocks=(1, 1, 1, 2), global_attn=False, attn_heads=8, wavelet=False):
        super().__init__()
        n_levels = len(enc_blocks)
        widths = [width * 2 ** i for i in range(n_levels + 1)]
        if global_attn:
            widths[-1] = max(attn_heads, widths[-1] * 2 // 3 // attn_heads * attn_heads)

        def block(level, ch):
            if global_attn and level == n_levels:
                return TransformerBlock(ch, attn_heads, 2.0, attn="spatial")
            return NAFBlock(ch)

        self.wavelet = wavelet
        io_ch = 12 if wavelet else 3
        self.net = RestorationUNet(block, in_ch=io_ch, out_ch=io_ch, widths=widths,
                                   enc_blocks=enc_blocks, middle_blocks=middle_blocks,
                                   dec_blocks=dec_blocks, predict_residual=True)

    def forward(self, x):
        if self.wavelet:
            return haar_iwt(self.net(haar_dwt(x)))
        return self.net(x)


def build_model(model_cfg):
    m = dict(model_cfg or {})
    return PlanNet(width=m.get("width", 24),
                   enc_blocks=tuple(m.get("enc_blocks", [2, 2, 2, 4])),
                   middle_blocks=m.get("middle_blocks", 3),
                   dec_blocks=tuple(m.get("dec_blocks", [1, 1, 1, 2])),
                   global_attn=m.get("global_attn", False),
                   attn_heads=m.get("attn_heads", 8),
                   wavelet=m.get("wavelet", False))
