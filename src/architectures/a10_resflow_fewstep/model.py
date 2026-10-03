"""
a10_resflow_fewstep -- the explicitly generative candidate: a residual flow
from the corrupted image to the clean one, run in very few steps.

a04 (round 1) diffused from pure noise and collapsed. This model never
starts from noise. Following InDI (Delbracio & Milanfar, 2023) and
rectified flow, it defines a straight path between the two images,

        x_t = (1 - t) * clean + t * corrupted,        t in [0, 1]

and trains one time-conditioned network to predict `clean` from any point
x_t on that path (it also always sees the original corrupted image).
Inference walks the path backwards in K steps: predict clean, step part of
the way towards it, predict again. K=1 is plain regression; K=2 lets the
second pass refine the first.

One addition for few-step use: half of the training batches build x_t from
the network's OWN first prediction instead of the true clean image, so the
second step is trained on exactly the input it will meet at inference.

FLOPs: the competition measures one call of the submitted model, and that
call runs all K passes, so each pass gets ~1/K of the 5 GFLOPs budget
(a smaller a05-style body). Expect better DISTS than the single-pass models
if it works, and somewhat lower PSNR.
"""
import torch
import torch.nn as nn

from src.common.layers import NAFBlock, sinusoidal_time_embedding
from src.common.unet import RestorationUNet
from src.common.bundle import RestorationBundle

NAME = "a10_resflow_fewstep"


class ResFlowNet(nn.Module):
    """One pass: (x_t, corrupted, t) -> estimate of the clean image."""

    def __init__(self, width=20, enc_blocks=(1, 1, 2, 2), middle_blocks=2,
                 dec_blocks=(1, 1, 1, 1), time_dim=64):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim * 2), nn.SiLU(),
                                      nn.Linear(time_dim * 2, time_dim))
        widths = [width * 2 ** i for i in range(len(enc_blocks) + 1)]
        # 6 input channels = x_t and the corrupted image; the output is a
        # correction added to x_t (the first 3 input channels).
        self.unet = RestorationUNet(lambda level, ch: NAFBlock(ch), in_ch=6, out_ch=3,
                                    widths=widths, enc_blocks=enc_blocks,
                                    middle_blocks=middle_blocks, dec_blocks=dec_blocks,
                                    cond_dim=time_dim, predict_residual=True)

    def forward(self, x_t, corrupted, t):
        emb = self.time_mlp(sinusoidal_time_embedding(t * 1000.0, self.time_dim))
        return self.unet(torch.cat([x_t, corrupted], dim=1), emb.to(x_t.dtype))


class ResFlowSampler(nn.Module):
    """The inference model: runs all K passes inside one forward call, so a FLOPs
    counter applied to it reports the true total cost."""

    def __init__(self, net: ResFlowNet, steps=2):
        super().__init__()
        self.net = net
        self.steps = steps

    def forward(self, corrupted):
        x = corrupted
        for i in range(self.steps):
            t = corrupted.new_full((corrupted.shape[0],), 1.0 - i / self.steps)
            estimate = self.net(x, corrupted, t)
            t_next = 1.0 - (i + 1) / self.steps
            x = (1.0 - t_next) * estimate + t_next * corrupted
        return estimate


class ResFlowBundle(RestorationBundle):
    def __init__(self, cfg, device, model):
        super().__init__(cfg, device, model, name=NAME)
        self.self_pred_prob = cfg["train"].get("self_pred_prob", 0.5)

    def forward_train(self, corrupted, clean, labels):
        net = self.model.net
        b = clean.shape[0]
        ones = clean.new_ones(b)
        # Half the samples sit at t=1 (the first step, always taken at inference);
        # the rest are spread over the path.
        t = torch.where(torch.rand(b, device=clean.device) < 0.5, ones,
                        torch.rand(b, device=clean.device))
        source = clean
        if torch.rand(()).item() < self.self_pred_prob:
            with torch.no_grad():
                source = net(corrupted, corrupted, ones).float().clamp(0, 1)
        tt = t.view(-1, 1, 1, 1)
        x_t = (1.0 - tt) * source + tt * corrupted
        return net(x_t, corrupted, t), None, {}


def build_model(model_cfg):
    m = dict(model_cfg or {})
    net = ResFlowNet(width=m.get("width", 20),
                     enc_blocks=tuple(m.get("enc_blocks", [1, 1, 2, 2])),
                     middle_blocks=m.get("middle_blocks", 2),
                     dec_blocks=tuple(m.get("dec_blocks", [1, 1, 1, 1])),
                     time_dim=m.get("time_dim", 64))
    return ResFlowSampler(net, steps=m.get("steps", 2))


def build(cfg, device):
    return ResFlowBundle(cfg, device, build_model(cfg.get("model")))
