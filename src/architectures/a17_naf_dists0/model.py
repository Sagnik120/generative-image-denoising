"""
a17_naf_dists0 -- a05 with the DISTS loss on from step 0 instead of from the halfway point.

The network is src/common/plannet.py::PlanNet; only config.yaml differs
between the round-3 architectures (see docs/IMPLEMENTATION_PLAN.md).
"""
from src.common.plannet import PlanNet, build_model
from src.common.bundle import RestorationBundle

NAME = "a17_naf_dists0"


def build(cfg, device):
    return RestorationBundle(cfg, device, build_model(cfg.get("model")), name=NAME)
