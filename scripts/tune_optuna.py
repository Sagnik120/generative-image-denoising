#!/usr/bin/env python3
"""
Hyperparameter tuning of the final architecture (a29_gan_decnoise) with Optuna.
Plan and reasoning: docs/HYPERPARAMETER_TUNING.md.

Each trial trains a29 from scratch for --trial_steps (default 10k = 4 epochs of
2.5k) with a complete, compressed cosine schedule, then scores it on a
132-image subset of benchmark v1 (4 per case) at noise x0.5:

    J = PSNR + 20 * SSIM - 20 * DISTS        (maximised)

Constraints (trial marked FAILED): diversity < 0.6 grey levels at noise x0.5,
final training sample spread > 3 x the diversity target, non-finite losses.

Search space (5): learning rate, discriminator/generator lr ratio,
adversarial weight, diversity target, SSIM weight. Sampler: TPE; pruner:
successive halving with rungs at 2.5k and 5k steps. Trial 0 is the current
a29 setting. Several workers (one per GPU) share one SQLite study.

Usage (one worker per GPU, same study):
    CUDA_VISIBLE_DEVICES=0 python scripts/tune_optuna.py --data_root <DATA> --enqueue_baseline
    CUDA_VISIBLE_DEVICES=1 python scripts/tune_optuna.py --data_root <DATA>
After the search (writes the phase-B and final configs from the best trial):
    python scripts/tune_optuna.py --write_configs
Summary of all trials:
    python scripts/tune_optuna.py --summary
"""
import argparse
import copy
import csv
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import optuna
import torch
import yaml

from src.registry import build_bundle, default_config_path
from src.common.utils import get_device, load_yaml, set_seed
from src.common.dataset import BenchmarkDataset, build_dataloaders
from src.common.metrics import evaluate_batch_per_image
from src.common.gennet import reseed_noise, set_noise_scale

BASE_ARCH = "a29_gan_decnoise"
OUT_DIR = PROJECT_ROOT / "results" / "optuna"
BASELINE = {"lr": 4e-4, "disc_lr_ratio": 1.0, "adv_weight": 0.005, "div_target": 1.5,
            "ssim_weight": 0.2}


def parse_args():
    p = argparse.ArgumentParser(description="Optuna tuning of a29_gan_decnoise.")
    p.add_argument("--data_root", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--study_name", default="a29_tune")
    p.add_argument("--storage", default=f"sqlite:///{OUT_DIR / 'a29_tune.db'}")
    p.add_argument("--n_trials", type=int, default=40, help="Total cap over all workers.")
    p.add_argument("--timeout_hours", type=float, default=14.0, help="Per worker.")
    p.add_argument("--trial_steps", type=int, default=10000)
    p.add_argument("--rungs", nargs="*", type=int, default=[2500, 5000])
    p.add_argument("--noise_scale", type=float, default=0.5)
    p.add_argument("--min_diversity", type=float, default=0.6)
    p.add_argument("--enqueue_baseline", action="store_true",
                   help="Add the current a29 setting as the first trial (one worker only).")
    p.add_argument("--write_configs", action="store_true")
    p.add_argument("--summary", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Tiny trials (60 steps) to test the setup.")
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    return p.parse_args()


def suggest(trial):
    return {
        "lr": trial.suggest_float("lr", 2e-4, 8e-4, log=True),
        "disc_lr_ratio": trial.suggest_categorical("disc_lr_ratio", [0.5, 1.0, 2.0]),
        "adv_weight": trial.suggest_float("adv_weight", 0.0025, 0.01, log=True),
        "div_target": trial.suggest_float("div_target", 1.0, 1.8),
        "ssim_weight": trial.suggest_float("ssim_weight", 0.1, 0.35),
    }


def apply_params(cfg, hp):
    t = cfg["train"]
    t["lr"] = float(hp["lr"])
    t["gan"]["disc_lr_ratio"] = float(hp["disc_lr_ratio"])
    t["gan"]["adv_weight"] = float(hp["adv_weight"])
    t["stochastic"]["div_target"] = float(hp["div_target"])
    t["loss"]["ssim"] = float(hp["ssim_weight"])
    return cfg


def subset_indices(ds, per_case=(2, 1, 1)):
    """4 images per case: 2 natural, 1 low-light, 1 medical (132 for benchmark v1)."""
    want = dict(zip(("natural", "lowlight", "medical"), per_case))
    taken, idx = {}, []
    for i, m in enumerate(ds.meta):
        key = (m["case"], m["domain"])
        if taken.get(key, 0) < want.get(m["domain"], 0):
            taken[key] = taken.get(key, 0) + 1
            idx.append(i)
    return idx


@torch.no_grad()
def score(model, loader, device, noise_scale, div_images=16, div_seeds=4):
    model.eval()
    set_noise_scale(model, noise_scale)
    reseed_noise(model, 0)
    sums, n = {"psnr": 0.0, "ssim": 0.0, "dists": 0.0}, 0
    for batch in loader:
        corrupted, clean = batch[0].to(device), batch[1].to(device)
        m = evaluate_batch_per_image(model(corrupted).float().clamp(0, 1), clean, device)
        for k in sums:
            sums[k] += sum(v for v in m[k] if v == v)
        n += corrupted.size(0)
    res = {k: v / max(n, 1) for k, v in sums.items()}
    # diversity: mean per-pixel std (grey levels) over seeds, on the first images
    corrupted = next(iter(loader))[0][:div_images].to(device)
    outs = []
    for seed in range(div_seeds):
        reseed_noise(model, seed)
        outs.append((model(corrupted).float().clamp(0, 1) * 255.0).round())
    res["diversity"] = torch.stack(outs).std(dim=0).mean().item()
    reseed_noise(model, 0)
    set_noise_scale(model, 1.0)
    res["J"] = res["psnr"] + 20.0 * res["ssim"] - 20.0 * res["dists"]
    return res


class Objective:
    def __init__(self, args, device):
        self.args, self.device = args, device
        self.base_cfg = load_yaml(PROJECT_ROOT / default_config_path(BASE_ARCH))
        steps = 60 if args.smoke else args.trial_steps
        self.steps = steps
        self.rungs = [r for r in ([20, 40] if args.smoke else args.rungs) if r < steps]
        cfg = self._trial_cfg(BASELINE)
        self.train_loader, _ = build_dataloaders(args.data_root, cfg)
        ds = BenchmarkDataset(args.data_root)
        sub = torch.utils.data.Subset(ds, subset_indices(ds))
        self.val_loader = torch.utils.data.DataLoader(sub, batch_size=16, shuffle=False, num_workers=4)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = OUT_DIR / "trials.csv"

    def _trial_cfg(self, hp):
        cfg = apply_params(copy.deepcopy(self.base_cfg), hp)
        spe = cfg["data"].get("steps_per_epoch", 2500)
        cfg["train"]["epochs"] = max(1, math.ceil(self.steps / spe))
        cfg["data"]["steps_per_epoch"] = min(spe, self.steps)
        cfg["train"]["warmup_steps"] = min(500, max(1, self.steps // 20))
        cfg["train"]["gan"]["ramp_steps"] = min(1000, max(1, self.steps // 10))
        return cfg

    def _log(self, row):
        new = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def __call__(self, trial):
        hp = suggest(trial)
        cfg = self._trial_cfg(hp)
        set_seed(cfg["train"].get("seed", 42))
        bundle = build_bundle(BASE_ARCH, cfg, self.device)
        t0, step, spreads, res = time.time(), 0, [], None
        status = "complete"
        try:
            while step < self.steps:
                for batch in self.train_loader:
                    out = bundle.train_step(batch[0].to(self.device, non_blocking=True),
                                            batch[1].to(self.device, non_blocking=True))
                    step += 1
                    if not math.isfinite(out.get("loss", float("nan"))) and bundle.bad_steps > 50:
                        status = "failed: non-finite losses"
                        return float("nan")
                    if "spread255" in out:
                        spreads = (spreads + [out["spread255"]])[-500:]
                    if step in self.rungs:
                        res = score(bundle.get_inference_model(), self.val_loader, self.device,
                                    self.args.noise_scale)
                        trial.report(res["J"], step)
                        print(f"[trial {trial.number}] step {step}: J {res['J']:.3f} "
                              f"PSNR {res['psnr']:.3f} SSIM {res['ssim']:.4f} DISTS {res['dists']:.4f} "
                              f"div {res['diversity']:.2f}", flush=True)
                        if trial.should_prune():
                            status = f"pruned at {step}"
                            raise optuna.TrialPruned()
                    if step >= self.steps:
                        break
            res = score(bundle.get_inference_model(), self.val_loader, self.device, self.args.noise_scale)
            spread = float(np.mean(spreads)) if spreads else float("nan")
            for k, v in {**res, "train_spread255": spread}.items():
                trial.set_user_attr(k, v)
            if res["diversity"] < self.args.min_diversity:
                status = f"failed: diversity {res['diversity']:.2f}"
                return float("nan")
            if spread > 3.0 * hp["div_target"]:
                status = f"failed: spread {spread:.2f}"
                return float("nan")
            return res["J"]
        finally:
            r = res or {}
            self._log({"trial": trial.number, "status": status, "steps": step,
                       "minutes": round((time.time() - t0) / 60, 1), **hp,
                       **{k: round(r.get(k, float("nan")), 4) for k in ("J", "psnr", "ssim", "dists", "diversity")}})
            del bundle
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


def load_study(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=args.study_name, storage=args.storage, direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(n_startup_trials=6, multivariate=True),
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=args.rungs[0], reduction_factor=3))


def write_configs(study):
    best = study.best_trial
    hp = {**BASELINE, **best.params}
    print(f"[tune] best trial {best.number}: J {best.value:.3f}  {hp}  {best.user_attrs}")
    base = load_yaml(PROJECT_ROOT / default_config_path(BASE_ARCH))
    for arch, epochs, note in (("a29_tuned_30k", 12, "phase B: best Optuna setting, 30k steps"),
                               ("a29_tuned", 80, "final: best Optuna setting, 200k steps")):
        cfg = apply_params(copy.deepcopy(base), hp)
        cfg["architecture"] = arch
        cfg["train"]["epochs"] = epochs
        path = PROJECT_ROOT / default_config_path(arch)
        header = (f"# Config for {arch}. Round 8 ({note}); written by scripts/tune_optuna.py "
                  f"from trial {best.number} (J {best.value:.3f}).\n# Tuned: {hp}\n")
        path.write_text(header + yaml.safe_dump(cfg, sort_keys=False))
        print(f"[tune] wrote {path}")


def summary(study):
    rows = sorted([t for t in study.trials if t.value is not None and t.value == t.value],
                  key=lambda t: -t.value)
    states = {}
    for t in study.trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1
    print(f"[tune] {len(study.trials)} trials: {states}")
    for t in rows[:10]:
        ua = t.user_attrs
        print(f"  #{t.number:3d} J {t.value:.3f}  PSNR {ua.get('psnr', 0):.3f} SSIM {ua.get('ssim', 0):.4f} "
              f"DISTS {ua.get('dists', 0):.4f} div {ua.get('diversity', 0):.2f}  {t.params}")


def main():
    args = parse_args()
    study = load_study(args)
    if args.summary:
        return summary(study)
    if args.write_configs:
        return write_configs(study)
    if args.enqueue_baseline and not study.trials:
        study.enqueue_trial(BASELINE)
    device = get_device(args.device)
    objective = Objective(args, device)
    cap = optuna.study.MaxTrialsCallback(
        args.n_trials, states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
                               optuna.trial.TrialState.FAIL, optuna.trial.TrialState.RUNNING))
    study.optimize(objective, timeout=args.timeout_hours * 3600, callbacks=[cap],
                   catch=(RuntimeError,))
    summary(study)


if __name__ == "__main__":
    main()
