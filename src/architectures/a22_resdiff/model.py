"""
a22_resdiff -- residual diffusion (ResShift-style), full-size network, 1 sampling step. Evaluate the same checkpoint at 2 and 4 steps with scripts/eval_generative.py.

Round 4 (strictly generative, see docs/PLAN.md). Network, sampler and
training are in src/common/resdiff.py; only config.yaml differs between the
diffusion models. The sampler runs every step inside one forward call.
"""
from src.common.resdiff import ResDiffSampler, ResDiffBundle, build_sampler as build_model

NAME = "a22_resdiff"


def build(cfg, device):
    return ResDiffBundle(cfg, device, build_model(cfg.get("model")), name=NAME)
