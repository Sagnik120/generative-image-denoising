"""
a14_wavelet_w28 -- a09 (Haar-domain NAF U-Net) at width 28 (about 2.5 GFLOPs). Size study: is the
wavelet front end the better way to be small?

The network is src/common/plannet.py::PlanNet; only config.yaml differs
between the round-3 architectures (see docs/IMPLEMENTATION_PLAN.md).
"""
from src.common.plannet import PlanNet, build_model
from src.common.bundle import RestorationBundle

NAME = "a14_wavelet_w28"


def build(cfg, device):
    return RestorationBundle(cfg, device, build_model(cfg.get("model")), name=NAME)
