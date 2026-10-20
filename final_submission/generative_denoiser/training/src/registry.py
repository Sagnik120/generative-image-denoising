"""Model registry: the generative denoiser."""
import importlib

ARCHITECTURES = {"generative_denoiser": "src.architectures.generative_denoiser.model"}
ROUND4 = list(ARCHITECTURES)


def get_architecture_module(name):
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{name}'. Available: {list(ARCHITECTURES)}")
    return importlib.import_module(ARCHITECTURES[name])


def build_bundle(name, cfg, device):
    return get_architecture_module(name).build(cfg, device)


def default_config_path(name):
    return ARCHITECTURES[name].replace(".model", "").replace(".", "/") + "/config.yaml"
