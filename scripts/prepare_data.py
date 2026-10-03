#!/usr/bin/env python3
"""
One-time data preparation. Run this ONCE on the training machine before any
training run; it is safe to re-run (finished datasets are skipped and
partial downloads resume).

  1. Downloads every dataset in src/common/dataset.py::SOURCES (~30 GB),
     trying each mirror in turn.
  2. Converts them into the cached training views and writes manifest.json.
  3. Builds the fixed benchmark used to validate every architecture.

It exits with an error if ANY dataset is incomplete, so a training run can
never silently start on less data than intended.

Usage:
    python scripts/prepare_data.py --data_root /path/to/data
    python scripts/prepare_data.py --check_urls          # no download, just test every mirror
    python scripts/prepare_data.py --data_root ... --only lol bsds500
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import requests

from src.common.dataset import (SOURCES, TRAIN_VIEWS, HOLDOUT_VIEWS, ensure_datasets,
                                build_benchmark, read_view)


def check_urls() -> bool:
    """Requests the first KB of every mirror. A source passes if any mirror answers."""
    all_ok = True
    for name, spec in SOURCES.items():
        source_ok = False
        for url, _ in spec["mirrors"]:
            try:
                r = requests.get(url, headers={"Range": "bytes=0-1023"}, stream=True, timeout=45)
                ok = r.status_code in (200, 206)
                total = r.headers.get("content-range", "").split("/")[-1]
                size = f"{int(total) / 1e9:.2f} GB" if total.isdigit() else "size unknown"
                r.close()
            except Exception as e:
                ok, size = False, str(e)[:60]
            source_ok |= ok
            print(f"  {'OK  ' if ok else 'FAIL'} {name:14s} {size:14s} {url}")
        if not source_ok:
            all_ok = False
            print(f"  >>> {name}: NO WORKING MIRROR")
    return all_ok


def main():
    p = argparse.ArgumentParser(description="Download + cache all datasets and build the benchmark.")
    p.add_argument("--data_root", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--only", nargs="*", default=None, choices=list(SOURCES),
                   help="Prepare just these sources (default: all).")
    p.add_argument("--check_urls", action="store_true",
                   help="Only test that every dataset has a reachable mirror, then exit.")
    p.add_argument("--procs", type=int, default=None, help="Worker processes for caching.")
    p.add_argument("--skip_benchmark", action="store_true")
    p.add_argument("--rebuild_benchmark", action="store_true")
    args = p.parse_args()

    if args.check_urls:
        ok = check_urls()
        print("\nAll datasets reachable." if ok else "\nSome datasets have no working mirror.")
        sys.exit(0 if ok else 1)

    total_gb = sum(SOURCES[n]["gb"] for n in (args.only or SOURCES))
    print(f"[prepare_data] data_root={args.data_root}  (~{total_gb:.0f} GB to download if not cached)")
    ensure_datasets(args.data_root, only=args.only, strict=True, num_procs=args.procs)

    print("\n[prepare_data] cached views:")
    for view in list(TRAIN_VIEWS) + HOLDOUT_VIEWS:
        print(f"  {view:18s} {len(read_view(args.data_root, view)):7d} images")

    if not args.skip_benchmark and not args.only:
        build_benchmark(args.data_root, force=args.rebuild_benchmark)
    print("\n[prepare_data] DONE. Data is ready for training.")


if __name__ == "__main__":
    main()
