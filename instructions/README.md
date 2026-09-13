# instructions/ — How to Run Each Architecture

Practical, step-by-step "how do I actually run this" guides. For the
*reasoning* behind each architecture's design, see `docs/` instead.

## General workflow (applies to every architecture)

1. **Local quick check** (a few seconds, CPU is fine):
   ```bash
   python scripts/verify_all.py
   ```
   Confirms every architecture builds and runs a forward/backward pass
   correctly before you spend GPU time on anything.

2. **Train** (run this in Colab for GPU acceleration — see
   `notebooks/train_colab.ipynb`, or locally if you have a GPU):
   ```bash
   python scripts/train.py --arch <architecture_name> --data_root <path_to_data>
   ```

3. **Evaluate + package results**:
   ```bash
   python scripts/evaluate.py --arch <architecture_name> --data_root <path_to_data>
   ```
   Writes `results/<architecture_name>/metrics/final_evaluation_summary.json`
   and zips the whole `results/<architecture_name>/` folder.

4. **Export a submission-ready inference package**:
   ```bash
   python scripts/export_inference.py --arch <architecture_name>
   ```
   Produces `results/<architecture_name>/submission/` containing
   `model_weights.pt` + a standalone `inference.py` + `README.md` --
   exactly matching the competition's Section 7 submission requirements.

5. **After training multiple architectures, compare them**:
   ```bash
   python scripts/compare_architectures.py
   ```

## Per-architecture guides
- `nafnet_unet/README.md`
- `restormer_lite/README.md`
- `pix2pix_gan/README.md`
- `tiny_ddpm_sr3/README.md`

Each covers that architecture's specific config knobs and any
architecture-specific quirks in the training/evaluation commands.
