"""
Shared U-Net skeleton for the round-2 architectures (a05 onward).

Every round-2 model is this same encoder / bottleneck / decoder shell with
additive skip connections; the architectures differ only in WHICH block runs
at each resolution (and in what is wrapped around the shell). Keeping the
shell in one place means a difference in results between two architectures
comes from their blocks, not from an accidental difference in plumbing.
"""
import torch
import torch.nn as nn

from .layers import Downsample, Upsample, FiLM


class RestorationUNet(nn.Module):
    """
    widths:        channels per level; the last entry is the bottleneck.
    enc_blocks:    blocks per encoder level        (len == len(widths) - 1)
    dec_blocks:    blocks per decoder level, coarse -> fine (same length)
    middle_blocks: blocks in the bottleneck
    block_fn:      block_fn(level, channels) -> nn.Module; level 0 is full
                   resolution, level len(widths)-1 is the bottleneck.
    cond_dim:      if set, forward() takes a conditioning vector that scales
                   and shifts the features after every stage (FiLM).
    """

    def __init__(self, block_fn, in_ch=3, out_ch=3, widths=(24, 48, 96, 192),
                 enc_blocks=(2, 2, 4), middle_blocks=4, dec_blocks=(2, 2, 2),
                 cond_dim=None, predict_residual=True):
        super().__init__()
        assert len(enc_blocks) == len(dec_blocks) == len(widths) - 1
        self.predict_residual = predict_residual
        self.res_ch = min(in_ch, out_ch)
        n_levels = len(widths) - 1

        self.intro = nn.Conv2d(in_ch, widths[0], kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for lvl, n_blocks in enumerate(enc_blocks):
            self.encoders.append(nn.Sequential(
                *[block_fn(lvl, widths[lvl]) for _ in range(n_blocks)]))
            self.downs.append(Downsample(widths[lvl], widths[lvl + 1]))

        self.middle = nn.Sequential(
            *[block_fn(n_levels, widths[-1]) for _ in range(middle_blocks)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i, n_blocks in enumerate(dec_blocks):
            lvl = n_levels - 1 - i
            self.ups.append(Upsample(widths[lvl + 1], widths[lvl]))
            self.decoders.append(nn.Sequential(
                *[block_fn(lvl, widths[lvl]) for _ in range(n_blocks)]))

        self.outro = nn.Conv2d(widths[0], out_ch, kernel_size=3, padding=1)

        self.cond_dim = cond_dim
        if cond_dim is not None:
            enc_w = list(widths[:-1])
            self.enc_films = nn.ModuleList([FiLM(cond_dim, w) for w in enc_w])
            self.mid_film = FiLM(cond_dim, widths[-1])
            self.dec_films = nn.ModuleList([FiLM(cond_dim, w) for w in reversed(enc_w)])
            for film in [*self.enc_films, self.mid_film, *self.dec_films]:
                nn.init.zeros_(film.proj.weight)   # start as the unconditioned net
                nn.init.zeros_(film.proj.bias)

    def forward(self, x, cond=None):
        inp = x
        x = self.intro(x)

        skips = []
        for i, (enc, down) in enumerate(zip(self.encoders, self.downs)):
            x = enc(x)
            if cond is not None:
                x = self.enc_films[i](x, cond)
            skips.append(x)
            x = down(x)

        x = self.middle(x)
        if cond is not None:
            x = self.mid_film(x, cond)

        for i, (up, dec, skip) in enumerate(zip(self.ups, self.decoders, reversed(skips))):
            x = up(x) + skip
            x = dec(x)
            if cond is not None:
                x = self.dec_films[i](x, cond)

        out = self.outro(x)
        if self.predict_residual:
            out = out + inp[:, :self.res_ch]
        return out
