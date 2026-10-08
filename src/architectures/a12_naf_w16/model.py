"""
a12_naf_w16 -- a05 at width 16 (about 2.2 GFLOPs). Size study: what does halving the FLOPs cost?

The network is src/common/plannet.py::PlanNet; only config.yaml differs
between the round-3 architectures (see docs/IMPLEMENTATION_PLAN.md).
"""
from src.common.plannet import PlanNet, build_model
from src.common.bundle import RestorationBundle

NAME = "a12_naf_w16"


def build(cfg, device):
    return RestorationBundle(cfg, device, build_model(cfg.get("model")), name=NAME)
