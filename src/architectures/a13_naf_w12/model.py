"""
a13_naf_w12 -- a05 at width 12 (about 1.3 GFLOPs). Size study: the small end.

The network is src/common/plannet.py::PlanNet; only config.yaml differs
between the round-3 architectures (see docs/IMPLEMENTATION_PLAN.md).
"""
from src.common.plannet import PlanNet, build_model
from src.common.bundle import RestorationBundle

NAME = "a13_naf_w12"


def build(cfg, device):
    return RestorationBundle(cfg, device, build_model(cfg.get("model")), name=NAME)
