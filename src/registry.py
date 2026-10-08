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

Round-2 architectures also expose `build_model(model_cfg) -> nn.Module` (the
bare inference network), which scripts/export_inference.py uses to rebuild
the model from its saved config.

Nothing else needs to change -- the Trainer, dataset pipeline, metrics, and
Colab notebook are all architecture-agnostic.
"""
import importlib

ARCHITECTURES = {
    "a01_nafnet_unet": "src.architectures.a01_nafnet_unet.model",
    "a02_restormer_lite": "src.architectures.a02_restormer_lite.model",
    "a03_pix2pix_gan": "src.architectures.a03_pix2pix_gan.model",
    "a04_tiny_ddpm_sr3": "src.architectures.a04_tiny_ddpm_sr3.model",
    # ---- round 2: new data / degradation / loss pipeline, ~5 GFLOPs each ----
    "a05_nafnet_v2": "src.architectures.a05_nafnet_v2.model",
    "a06_restormer_v2": "src.architectures.a06_restormer_v2.model",
    "a07_naf_attn_hybrid": "src.architectures.a07_naf_attn_hybrid.model",
    "a08_degradation_aware": "src.architectures.a08_degradation_aware.model",
    "a09_wavelet_naf": "src.architectures.a09_wavelet_naf.model",
    "a10_resflow_fewstep": "src.architectures.a10_resflow_fewstep.model",
    "a11_freq_spatial": "src.architectures.a11_freq_spatial.model",
    # ---- round 3: single-change experiments on a05, then the final model ----
    # (see docs/IMPLEMENTATION_PLAN.md; all are src/common/plannet.py::PlanNet)
    "a12_naf_w16": "src.architectures.a12_naf_w16.model",
    "a13_naf_w12": "src.architectures.a13_naf_w12.model",
    "a14_wavelet_w28": "src.architectures.a14_wavelet_w28.model",
    "a15_naf_degv2": "src.architectures.a15_naf_degv2.model",
    "a16_naf_attn_degv2": "src.architectures.a16_naf_attn_degv2.model",
    "a17_naf_dists0": "src.architectures.a17_naf_dists0.model",
    "a18_naf_dists0_strong": "src.architectures.a18_naf_dists0_strong.model",
    "a19_final": "src.architectures.a19_final.model",
}

# Architectures are numbered in the order they were added. Round 1 (a01-a04)
# was trained on the original DIV2K-only pipeline with a different, noisy
# validation set, so its numbers are NOT comparable with round 2 onward.
ROUND1 = ["a01_nafnet_unet", "a02_restormer_lite", "a03_pix2pix_gan", "a04_tiny_ddpm_sr3"]
FINAL = "a19_final"              # its config.yaml is written by scripts/run_plan.py
ROUND3 = [name for name in ARCHITECTURES if "a12" <= name[:3] <= "a18"]
ROUND2 = [name for name in ARCHITECTURES if name not in ROUND1 + ROUND3 + [FINAL]]


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
