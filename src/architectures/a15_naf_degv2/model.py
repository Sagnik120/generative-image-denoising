"""
a15_naf_degv2 -- a05 trained on the v2 degradation pipeline. Compare with a05 (same network,
v1 pipeline) on benchmark v2's unseen families.

The network is src/common/plannet.py::PlanNet; only config.yaml differs
between the round-3 architectures (see docs/IMPLEMENTATION_PLAN.md).
"""
from src.common.plannet import PlanNet, build_model
from src.common.bundle import RestorationBundle

NAME = "a15_naf_degv2"


def build(cfg, device):
    return RestorationBundle(cfg, device, build_model(cfg.get("model")), name=NAME)
