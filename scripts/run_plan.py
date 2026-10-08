#!/usr/bin/env python3
"""
Runs the whole round-3 plan (docs/IMPLEMENTATION_PLAN.md) with ONE command:

    nohup python scripts/run_plan.py --data_root /path/to/data > run_plan.log 2>&1 &

What it does, in order:
  1. Builds benchmark v2 (new unseen families) if it is missing. CPU only, no download.
  2. Dry run of every round-3 architecture (30 steps + evaluate + export each).
     Stops here if any of them fails, before any real GPU time is spent.
  3. Re-evaluates the reference (a05) so it also has benchmark-v2 numbers.
  4. Trains, evaluates and exports the seven experiments (scripts/run_all.py,
     one per GPU in parallel).
  5. Applies the decision rules, writes results/plan/decision.{json,md} and the
     config of the final model (src/architectures/a19_final/config.yaml).
  6. Trains, evaluates and exports the final model.

Safe to re-run after an interruption: finished runs are skipped, interrupted
ones resume. Once the final model has started training its config is kept
(pass --redecide to recompute it; then delete results/a19_final first).

    --decide_only   only do step 5 and print the decision (no GPU needed)
    --skip_final    stop after step 5
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.registry import ROUND3, FINAL, default_config_path
from src.common.utils import load_yaml

REF = "a05_nafnet_v2"
SIZE_CANDIDATES = ["a12_naf_w16", "a13_naf_w12", "a14_wavelet_w28"]
DEG, ATTN = "a15_naf_degv2", "a16_naf_attn_degv2"
LOSS_CANDIDATES = ["a17_naf_dists0", "a18_naf_dists0_strong"]


def log(msg):
    print(f"[run_plan {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_result(results_root, arch):
    """Headline numbers of a finished run, or None."""
    path = Path(results_root) / arch / "metrics" / "final_evaluation_summary.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    unseen_v2 = d.get("benchmark_v2", {}).get("by_group", {}).get("unseen_v2", {}).get("psnr")
    return {"psnr": d["final_val_psnr"], "ssim": d["final_val_ssim"], "dists": d["final_val_dists"],
            "gflops": d["gflops"], "unseen_v2": unseen_v2}


def decide(results_root, final_epochs):
    """Applies the plan's decision rules. Returns (final config dict, report lines)."""
    R = {a: load_result(results_root, a) for a in [REF] + ROUND3}
    ref = R[REF]
    if ref is None:
        raise SystemExit(f"[run_plan] {REF} has no final_evaluation_summary.json under {results_root}; "
                         f"it is the reference every rule compares against.")
    notes = []

    # size: the smallest model that stays close to the reference on all three quality axes
    size = REF
    for a in SIZE_CANDIDATES:
        r = R[a]
        if r is None:
            notes.append(f"size: {a} has no result, ignored")
            continue
        close = (r["psnr"] >= ref["psnr"] - 0.15 and r["ssim"] >= ref["ssim"] - 0.004
                 and r["dists"] <= ref["dists"] + 0.005)
        notes.append(f"size: {a} {r['gflops']:.2f} GFLOPs, PSNR {r['psnr'] - ref['psnr']:+.3f}, "
                     f"SSIM {r['ssim'] - ref['ssim']:+.4f}, DISTS {r['dists'] - ref['dists']:+.4f} "
                     f"vs {REF} -> {'within tolerance' if close else 'too much quality lost'}")
        if close and r["gflops"] < R[size]["gflops"]:
            size = a
    notes.append(f"size: chosen body = {size}")

    # degradation pipeline v2
    use_deg, r = False, R[DEG]
    if r is None or r["unseen_v2"] is None or ref["unseen_v2"] is None:
        notes.append(f"degradation v2: no benchmark-v2 numbers for {DEG} or {REF} -> rejected")
    else:
        gain, drop = r["unseen_v2"] - ref["unseen_v2"], ref["psnr"] - r["psnr"]
        use_deg = gain >= 0.10 and drop <= 0.10
        notes.append(f"degradation v2: unseen-v2 PSNR {gain:+.3f}, overall PSNR {-drop:+.3f} vs {REF} "
                     f"-> {'accepted' if use_deg else 'rejected'}")

    # global-attention bottleneck (a16 vs a15: same data, same body otherwise)
    use_attn, a, b = False, R[ATTN], R[DEG]
    if a is None or b is None or a["unseen_v2"] is None or b["unseen_v2"] is None:
        notes.append(f"attention: no benchmark-v2 numbers for {ATTN} or {DEG} -> rejected")
    else:
        gain, drop = a["unseen_v2"] - b["unseen_v2"], b["psnr"] - a["psnr"]
        use_attn = gain >= 0.15 and drop <= 0.05
        notes.append(f"attention: unseen-v2 PSNR {gain:+.3f}, overall PSNR {-drop:+.3f} vs {DEG} "
                     f"-> {'accepted' if use_attn else 'rejected'}")

    # loss weighting
    loss_arch = None
    for a in LOSS_CANDIDATES:
        r = R[a]
        if r is None:
            notes.append(f"loss: {a} has no result, ignored")
            continue
        ok = r["dists"] <= ref["dists"] - 0.005 and r["psnr"] >= ref["psnr"] - 0.05
        notes.append(f"loss: {a} DISTS {r['dists'] - ref['dists']:+.4f}, PSNR {r['psnr'] - ref['psnr']:+.3f} "
                     f"vs {REF} -> {'acceptable' if ok else 'rejected'}")
        if ok and (loss_arch is None or r["dists"] < R[loss_arch]["dists"]):
            loss_arch = a
    notes.append(f"loss: chosen = {loss_arch or 'round-2 default'}")

    def cfg_of(arch):
        return load_yaml(PROJECT_ROOT / default_config_path(arch))

    cfg = cfg_of(REF)
    cfg["architecture"] = FINAL
    model = {k: v for k, v in cfg_of(size)["model"].items() if k != "predict_residual"}
    model["global_attn"] = use_attn
    cfg["model"] = model
    if use_deg:
        cfg["data"]["degradation"] = cfg_of(DEG)["data"]["degradation"]
    if loss_arch:
        cfg["train"]["loss"] = cfg_of(loss_arch)["train"]["loss"]
    cfg["train"]["epochs"] = final_epochs
    cfg["train"]["visualize_every"] = 8

    report = ["# Round-3 decision", "", f"Written by scripts/run_plan.py on {time.strftime('%Y-%m-%d %H:%M')}.", "",
              "| Arch | PSNR | SSIM | DISTS | GFLOPs | unseen-v2 PSNR |", "|---|---|---|---|---|---|"]
    for a, r in R.items():
        if r is None:
            report.append(f"| {a} | - | - | - | - | - |")
        else:
            u = "-" if r["unseen_v2"] is None else f"{r['unseen_v2']:.2f}"
            report.append(f"| {a} | {r['psnr']:.3f} | {r['ssim']:.4f} | {r['dists']:.4f} | {r['gflops']:.2f} | {u} |")
    report += ["", "## Rules applied", ""] + [f"- {n}" for n in notes]
    report += ["", f"## Final model ({FINAL})", "",
               f"- body: `{size}`; global attention: {use_attn}; degradation pipeline: {'v2' if use_deg else 'v1'}; "
               f"loss: `{loss_arch or 'round-2 default'}`; {final_epochs} epochs "
               f"({final_epochs * cfg['data'].get('steps_per_epoch', 2500):,} steps)"]
    decision = {"size": size, "global_attn": use_attn, "degradation_v2": use_deg,
                "loss": loss_arch, "final_epochs": final_epochs, "results": R}
    return cfg, decision, report


def write_decision(args):
    """Step 5. Returns False if the existing final config was kept."""
    import yaml
    out_dir = Path(args.results_root) / "plan"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, decision, report = decide(args.results_root, args.final_epochs)
    print("\n".join(report), flush=True)
    started = (Path(args.results_root) / FINAL / "checkpoints" / "last_model.pt").exists()
    if args.decide_only:
        return False
    if started and not args.redecide:
        log(f"{FINAL} has already started training: keeping its existing config.yaml "
            f"(use --redecide after deleting results/{FINAL} to change it).")
        return False
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2))
    (out_dir / "decision.md").write_text("\n".join(report) + "\n")
    config_path = PROJECT_ROOT / default_config_path(FINAL)
    config_path.write_text(f"# Config for {FINAL}. WRITTEN BY scripts/run_plan.py from the round-3 results;\n"
                           f"# the reasons are in results/plan/decision.md.\n"
                           + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None))
    log(f"final config written to {config_path}")
    return True


def run_all(args, archs, dry_run=False):
    cmd = [sys.executable, "scripts/run_all.py", "--data_root", args.data_root,
           "--results_root", args.results_root, "--archs", *archs,
           "--max_gpu_gb", str(args.max_gpu_gb)]
    if args.gpus:
        cmd += ["--gpus", *args.gpus]
    if args.allow_busy_gpu:
        cmd.append("--allow_busy_gpu")
    if dry_run:
        cmd.append("--dry_run")
    log("running: " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def reevaluate_reference(args):
    """Step 3: the reference was evaluated before benchmark v2 existed."""
    r = load_result(args.results_root, REF)
    if r is not None and r["unseen_v2"] is not None:
        return
    if not (Path(args.results_root) / REF / "checkpoints" / "best_model.pt").exists():
        log(f"WARNING: {REF} checkpoint not found; the degradation-v2 rule will be skipped.")
        return
    from run_all import find_gpus
    gpus = find_gpus(args.max_gpu_gb, args.gpus, args.allow_busy_gpu)
    if not gpus:
        raise SystemExit("[run_plan] no usable GPU found.")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus[0][0], "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    log(f"re-evaluating {REF} on benchmark v1 + v2")
    code = subprocess.call([sys.executable, "scripts/evaluate.py", "--arch", REF, "--data_root",
                            args.data_root, "--results_root", args.results_root],
                           cwd=str(PROJECT_ROOT), env=env)
    if code != 0:
        log(f"WARNING: re-evaluation of {REF} failed; the degradation-v2 rule will be skipped.")


def main():
    p = argparse.ArgumentParser(description="Run the round-3 plan end to end.")
    p.add_argument("--data_root", default=None)
    p.add_argument("--results_root", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--gpus", nargs="*", default=None)
    p.add_argument("--max_gpu_gb", type=float, default=30.0)
    p.add_argument("--allow_busy_gpu", action="store_true")
    p.add_argument("--final_epochs", type=int, default=80, help="x 2500 steps (80 = 200k steps).")
    p.add_argument("--skip_dry_run", action="store_true")
    p.add_argument("--skip_final", action="store_true")
    p.add_argument("--decide_only", action="store_true")
    p.add_argument("--redecide", action="store_true")
    args = p.parse_args()

    if args.decide_only:
        write_decision(args)
        return
    if not args.data_root:
        raise SystemExit("[run_plan] --data_root is required.")

    from src.common.dataset import build_benchmark
    from src.common.degradations import BENCHMARK_CASES_V2
    build_benchmark(args.data_root, seed=2027, version="v2", cases=BENCHMARK_CASES_V2)

    if not args.skip_dry_run:
        if run_all(args, ROUND3 + [FINAL], dry_run=True) != 0:
            raise SystemExit("[run_plan] dry run failed; fix it before the real runs "
                             "(logs: results/_dry_run/<arch>/logs/console.log).")
    reevaluate_reference(args)

    if run_all(args, ROUND3) != 0:
        log("WARNING: at least one experiment failed; deciding with the ones that finished.")
    write_decision(args)

    if args.skip_final:
        return
    code = run_all(args, [FINAL])
    final = load_result(args.results_root, FINAL)
    if final:
        log(f"FINAL {FINAL}: PSNR {final['psnr']:.3f}  SSIM {final['ssim']:.4f}  DISTS {final['dists']:.4f}  "
            f"{final['gflops']:.2f} GFLOPs. Submission: {Path(args.results_root) / FINAL / 'submission'}")
    sys.exit(code)


if __name__ == "__main__":
    main()
