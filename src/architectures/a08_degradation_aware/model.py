"""
a08_degradation_aware -- a07's body, told what is wrong with the image.

The brief's hardest requirement is one model for many corruption types AND
their combinations. A network with fixed weights must apply one compromise
filter to "heavy Gaussian noise", "mild motion blur + JPEG" and
"salt-and-pepper" alike. Here a tiny encoder first summarises the input's
corruption into a vector, and that vector rescales and shifts the features
of every stage (FiLM), so the same weights behave differently per input.

To make the vector actually describe the corruption, training adds an
auxiliary task: a linear head must predict, from the vector alone, which
degradation families were applied and how severely (the label the data
pipeline already knows). The head is training-only; inference never calls
it, so it costs no FLOPs. The encoder itself adds about 0.05 GFLOPs.

Compare against a07 (identical body, no conditioning) to read off what
degradation awareness is worth, especially on the pair / triple cases of the
benchmark.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.degradations import NUM_TRAIN_FAMILIES
from src.common.unet import RestorationUNet
from src.common.bundle import RestorationBundle
from src.architectures.a07_naf_attn_hybrid.model import hybrid_block_fn

NAME = "a08_degradation_aware"


class DegradationEncoder(nn.Module):
    """Four stride-2 convs, then the per-channel MEAN and STD over the image.
    The std matters: noise level is a spread, which average pooling alone hides."""

    def __init__(self, cond_dim=64, ch=(16, 32, 64, 64)):
        super().__init__()
        layers, prev = [], 3
        for c in ch:
            layers += [nn.Conv2d(prev, c, kernel_size=3, stride=2, padding=1), nn.GELU()]
            prev = c
        self.features = nn.Sequential(*layers)
        self.mlp = nn.Sequential(nn.Linear(prev * 2, cond_dim), nn.GELU(),
                                 nn.Linear(cond_dim, cond_dim))

    def forward(self, x):
        f = self.features(x).float().flatten(2)
        return self.mlp(torch.cat([f.mean(-1), f.std(-1)], dim=1))


class DegradationAwareNet(nn.Module):
    def __init__(self, widths=(24, 48, 96, 192, 256), enc_blocks=(2, 2, 2, 2),
                 middle_blocks=3, dec_blocks=(1, 1, 1, 2), cond_dim=64, predict_residual=True):
        super().__init__()
        self.encoder = DegradationEncoder(cond_dim)
        self.net = RestorationUNet(hybrid_block_fn(len(enc_blocks)), widths=widths,
                                   enc_blocks=enc_blocks, middle_blocks=middle_blocks,
                                   dec_blocks=dec_blocks, cond_dim=cond_dim,
                                   predict_residual=predict_residual)
        self.aux_head = nn.Linear(cond_dim, NUM_TRAIN_FAMILIES)     # training only

    def forward(self, x, return_aux=False):
        cond = self.encoder(x)
        out = self.net(x, cond.to(x.dtype))
        if return_aux:
            return out, self.aux_head(cond)
        return out


class DegradationAwareBundle(RestorationBundle):
    uses_labels = True

    def __init__(self, cfg, device, model):
        super().__init__(cfg, device, model, name=NAME)
        self.aux_weight = cfg["train"].get("aux_weight", 0.1)

    def forward_train(self, corrupted, clean, labels):
        pred, aux = self.model(corrupted, return_aux=True)
        if labels is None:
            return pred, None, {}
        aux_loss = F.smooth_l1_loss(aux.float(), labels.float())
        return pred, self.aux_weight * aux_loss, {"aux": aux_loss.item()}


def build_model(model_cfg):
    m = dict(model_cfg or {})
    return DegradationAwareNet(widths=tuple(m.get("widths", [24, 48, 96, 192, 256])),
                               enc_blocks=tuple(m.get("enc_blocks", [2, 2, 2, 2])),
                               middle_blocks=m.get("middle_blocks", 3),
                               dec_blocks=tuple(m.get("dec_blocks", [1, 1, 1, 2])),
                               cond_dim=m.get("cond_dim", 64),
                               predict_residual=m.get("predict_residual", True))


def build(cfg, device):
    return DegradationAwareBundle(cfg, device, build_model(cfg.get("model")))
