"""
a23_resdiff_2step -- residual diffusion at width 16 with 2 sampling steps (about 4.6 GFLOPs in total).

Round 4 (strictly generative, see docs/PLAN.md). Network, sampler and
training are in src/common/resdiff.py; only config.yaml differs between the
diffusion models. The sampler runs every step inside one forward call.
"""
from src.common.resdiff import ResDiffSampler, ResDiffBundle, build_sampler as build_model

NAME = "a23_resdiff_2step"


def build(cfg, device):
    return ResDiffBundle(cfg, device, build_model(cfg.get("model")), name=NAME)
