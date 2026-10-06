#!/usr/bin/env python3
"""
Trains, evaluates and exports several architectures with ONE command, using
every allowed GPU in parallel (one architecture per GPU at a time). When the
command exits, every GPU it used is free.

    nohup python scripts/run_all.py --data_root /path/to/data > run_all.log 2>&1 &

What it does:
  1. Checks the data (manifest + benchmark) BEFORE touching a GPU.
  2. Picks the GPUs: every GPU with at most --max_gpu_gb of memory (default
     30, so the 48 GB card is never used) that is currently idle. GPUs are
     addressed by UUID, so differing index orders between tools cannot send
     a job to the wrong card.
  3. Runs a queue: each GPU takes the next architecture and runs
     train -> evaluate -> export for it. Each architecture is its own
     process with its own log (results/<arch>/logs/console.log); if one
     fails, its GPU simply moves on to the next architecture.
  4. Finishes with scripts/compare_architectures.py and a status table.

Re-running the same command is safe: finished architectures are skipped and
interrupted ones resume from their last checkpoint. Use --dry_run first: it
rehearses every architecture (30 training steps, then the real evaluate and
export steps) and reports speed and peak GPU memory, without touching real
results.
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.registry import ARCHITECTURES, ROUND2, default_config_path
from src.common.utils import load_yaml

print_lock = threading.Lock()


def log(msg):
    with print_lock:
        print(f"[run_all {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def find_gpus(max_gb, requested, allow_busy):
    """Returns [(uuid, description)] of the GPUs to use."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used",
             "--format=csv,noheader,nounits"]).decode()
    except Exception as e:
        raise SystemExit(f"[run_all] nvidia-smi failed ({e}). This script needs NVIDIA GPUs; "
                         f"for a single run use scripts/train.py.")
    gpus = []
    for line in out.strip().splitlines():
        index, uuid, name, total, used = [x.strip() for x in line.split(",")]
        desc = f"GPU {index} {name} ({int(total) / 1024:.0f} GB, {int(used)} MB in use)"
        if requested is not None and index not in requested:
            continue
        if int(total) / 1024 > max_gb:
            log(f"skipping {desc}: larger than --max_gpu_gb {max_gb}")
            continue
        if int(used) > 1500 and not allow_busy:
            log(f"skipping {desc}: already in use (pass --allow_busy_gpu to use it anyway)")
            continue
        gpus.append((uuid, desc))
    return gpus


def is_finished(arch, results_root):
    """True if this architecture was trained to its last epoch, evaluated and exported."""
    results_dir = Path(results_root) / arch
    summary = results_dir / "metrics" / "final_evaluation_summary.json"
    log_csv = results_dir / "logs" / "training_log.csv"
    if not (summary.exists() and log_csv.exists()
            and (results_dir / "submission" / "model_weights.pt").exists()):
        return False
    epochs = load_yaml(PROJECT_ROOT / default_config_path(arch))["train"]["epochs"]
    last_line = log_csv.read_text().strip().splitlines()[-1]
    return last_line.split(",")[0] == str(epochs)


def run_arch(arch, gpu_uuid, args, status):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    results_root = Path(args.results_root) / "_dry_run" if args.dry_run else Path(args.results_root)
    log_path = results_root / arch / "logs" / "console.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    py, common = sys.executable, ["--arch", arch, "--data_root", args.data_root]

    # A dry run rehearses the WHOLE chain; its outputs live under results/_dry_run.
    device = ["--device", args.device] if args.device else []
    steps = [("train", [py, "scripts/train.py", *common, "--results_root", args.results_root,
                        "--resume", "auto", *device, *args.train_args.split()]
              + (["--dry_run"] if args.dry_run else [])),
             ("evaluate", [py, "scripts/evaluate.py", *common, *device,
                           "--results_root", str(results_root)]),
             ("export", [py, "scripts/export_inference.py", "--arch", arch, *device,
                         "--results_root", str(results_root)])]
    t0 = time.time()
    with open(log_path, "a") as f:
        for name, cmd in steps:
            f.write(f"\n===== {name}: {' '.join(cmd)} =====\n")
            f.flush()
            code = subprocess.call(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=f,
                                   stderr=subprocess.STDOUT)
            if code != 0:
                status[arch] = f"FAILED at {name} (exit {code}) -- see {log_path}"
                return
    status[arch] = f"OK ({(time.time() - t0) / 3600:.2f} h)"


def worker(gpu, jobs, args, status):
    uuid, desc = gpu
    while True:
        try:
            arch = jobs.get_nowait()
        except queue.Empty:
            return
        log(f"START {arch} on {desc}")
        try:
            run_arch(arch, uuid, args, status)
        except Exception as e:
            status[arch] = f"FAILED ({e})"
        log(f"END   {arch}: {status[arch]}")


def dry_run_figures(arch, results_root):
    """Pulls speed and peak memory out of a dry run's console log."""
    path = Path(results_root) / "_dry_run" / arch / "logs" / "console.log"
    if not path.exists():
        return ""
    lines = [l for l in path.read_text().splitlines() if "it/s  epoch_eta" in l]
    if not lines:
        return ""
    tail = lines[-1]
    rate = tail.split(" it/s")[0].split()[-1]
    mem = tail.split("gpu_mem=")[-1] if "gpu_mem=" in tail else "n/a (no GPU)"
    return f"  {rate} it/s, peak memory {mem}"


def main():
    p = argparse.ArgumentParser(description="Train + evaluate + export a queue of architectures.")
    p.add_argument("--data_root", required=True)
    p.add_argument("--results_root", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--archs", nargs="*", default=ROUND2, choices=list(ARCHITECTURES),
                   help="Architectures to run, in queue order (default: all of round 2).")
    p.add_argument("--gpus", nargs="*", default=None,
                   help="nvidia-smi GPU indices to use (default: every idle GPU within --max_gpu_gb).")
    p.add_argument("--max_gpu_gb", type=float, default=30.0)
    p.add_argument("--allow_busy_gpu", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                   help="30-step rehearsal of every architecture; real results are not touched.")
    p.add_argument("--force", action="store_true", help="Re-run architectures that already finished.")
    p.add_argument("--train_args", default="",
                   help='Extra arguments for every train.py call, e.g. --train_args "--epochs 30".')
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"],
                   help="Force a device for every step (default: auto-detect).")
    args = p.parse_args()

    # 1. data first: fail here, in seconds, rather than on the GPU.
    from src.common.dataset import check_training_data, benchmark_dir, load_manifest, SOURCES
    views = check_training_data(args.data_root)
    if not (benchmark_dir(args.data_root) / "meta.json").exists():
        raise SystemExit(f"[run_all] benchmark missing; run scripts/prepare_data.py --data_root {args.data_root}")
    bad = [k for k, v in load_manifest(args.data_root)["sources"].items() if not v.get("ok") and k in SOURCES]
    if bad:
        raise SystemExit(f"[run_all] datasets not ready: {bad}; re-run scripts/prepare_data.py")
    log("data OK: " + ", ".join(f"{k}={len(v)}" for k, v in views.items()))

    # 2. download the DISTS backbone once, so parallel runs do not race on it.
    subprocess.call([sys.executable, "-c", "from DISTS_pytorch import DISTS; DISTS()"],
                    cwd=str(PROJECT_ROOT))

    # 3. queue
    todo = []
    for arch in args.archs:
        if not args.dry_run and not args.force and is_finished(arch, args.results_root):
            log(f"skip {arch}: already trained, evaluated and exported (use --force to redo)")
        else:
            todo.append(arch)
    if not todo:
        log("nothing to do.")
        return

    gpus = find_gpus(args.max_gpu_gb, args.gpus, args.allow_busy_gpu)
    if not gpus:
        raise SystemExit("[run_all] no usable GPU found.")
    log(f"{len(todo)} architecture(s) on {len(gpus)} GPU(s): " + "; ".join(d for _, d in gpus))

    jobs, status = queue.Queue(), {}
    for arch in todo:
        jobs.put(arch)
    threads = []
    for gpu in gpus:
        t = threading.Thread(target=worker, args=(gpu, jobs, args, status))
        t.start()
        threads.append(t)
        time.sleep(20)          # stagger start-up
    for t in threads:
        t.join()

    # 4. summary
    if not args.dry_run:
        subprocess.call([sys.executable, "scripts/compare_architectures.py",
                         "--results_root", args.results_root], cwd=str(PROJECT_ROOT))
    print("\n" + "=" * 70 + f"\n{'DRY RUN ' if args.dry_run else ''}SUMMARY\n" + "=" * 70)
    for arch in todo:
        extra = dry_run_figures(arch, args.results_root) if args.dry_run else ""
        print(f"  {arch:26s} {status.get(arch, 'not run')}{extra}")
    print("=" * 70 + "\nALL DONE -- every GPU used by this command is now free.", flush=True)
    if any(not s.startswith("OK") for s in status.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
