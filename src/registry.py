"""
Central registry of all available architectures.

To add a NEW architecture later:
  1. Create src/architectures/<your_arch_name>/model.py exposing a
     `build(cfg, device)` function that returns a ModelBundle
     (see src/common/trainer.py's docstring for the required interface).
  2. Create src/architectures/<your_arch_name>/config.yaml.
  3. Add one line below: "<your_arch_name>": "src.architectures.<your_arch_name>.model"
  4. Create matching results/<your_arch_name>/{checkpoints,loss_curves,metrics,
     visualizations,logs}/ folders (or just run scripts/train.py once --
     it creates them automatically).
  5. (Optional but recommended) add instructions/<your_arch_name>/README.md
     and docs/<your_arch_name>/notes.md.

Nothing else needs to change -- the Trainer, dataset pipeline, metrics, and
Colab notebook are all architecture-agnostic.
"""
import importlib

ARCHITECTURES = {
    "nafnet_unet": "src.architectures.nafnet_unet.model",
    "restormer_lite": "src.architectures.restormer_lite.model",
    "pix2pix_gan": "src.architectures.pix2pix_gan.model",
    "tiny_ddpm_sr3": "src.architectures.tiny_ddpm_sr3.model",
}


def get_architecture_module(name: str):
    if name not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{name}'. Available: {list(ARCHITECTURES.keys())}"
        )
    return importlib.import_module(ARCHITECTURES[name])


def build_bundle(name: str, cfg: dict, device):
    module = get_architecture_module(name)
    return module.build(cfg, device)


def default_config_path(name: str) -> str:
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{name}'.")
    module_path = ARCHITECTURES[name]
    folder = module_path.replace(".model", "").replace(".", "/")
    return f"{folder}/config.yaml"
