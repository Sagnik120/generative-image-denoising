"""
a19_final -- the final model. scripts/run_plan.py rewrites config.yaml from the round-3
results (size, degradation pipeline, attention, loss) before training it; the
file in the repo is only the default (a05 + every round-3 change, 200k steps).

The network is src/common/plannet.py::PlanNet; only config.yaml differs
between the round-3 architectures (see docs/IMPLEMENTATION_PLAN.md).
"""
from src.common.plannet import PlanNet, build_model
from src.common.bundle import RestorationBundle

NAME = "a19_final"


def build(cfg, device):
    return RestorationBundle(cfg, device, build_model(cfg.get("model")), name=NAME)
