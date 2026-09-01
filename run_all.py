#!/usr/bin/env python3
"""Run the whole project end to end.

    python run_all.py                  # full analysis, 109 participants
    python run_all.py --synthetic 12   # offline smoke test, no network needed
    python run_all.py --subjects 20    # a faster real subset
    python run_all.py --skip-download  # reuse whatever is in data/raw

Raw data are never overwritten: download_data.py skips any file already on
disk, so re-running is cheap and safe.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402


def banner(step: int, total: int, title: str) -> float:
    print(f"\n{'=' * 72}\n[{step}/{total}] {title}\n{'=' * 72}")
    return time.time()


def done(t0: float) -> None:
    print(f"  ... {time.time() - t0:.1f}s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", type=int, default=cfg.N_SUBJECTS)
    ap.add_argument("--synthetic", type=int, nargs="?", const=12, default=None,
                    metavar="N")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-robustness", action="store_true",
                    help="skip the slowest stage (repeated re-extraction)")
    args = ap.parse_args(argv)

    total, step = 7, 0
    overall = time.time()

    if not args.skip_download:
        step += 1
        t = banner(step, total, "Acquiring data")
        from src import download_data
        argv_dl = (["--synthetic", str(args.synthetic)] if args.synthetic is not None
                   else ["--subjects", str(args.subjects)])
        if download_data.main(argv_dl) != 0 and args.synthetic is None:
            print("\nData acquisition incomplete. Fix this before continuing, or "
                  "run with --synthetic to smoke-test the pipeline offline.")
            return 1
        done(t)

    prov = cfg.read_provenance()
    if prov.get("synthetic"):
        print("\n" + "!" * 72)
        print("SYNTHETIC DATA. Output goes to figures/_smoke and results/_smoke.")
        print("Every number produced below is meaningless as a scientific result.")
        print("!" * 72)

    step += 1
    t = banner(step, total, "Extracting features")
    from src import feature_extraction
    feature_extraction.main()
    done(t)

    step += 1
    t = banner(step, total, "Statistical analysis")
    from src import statistics
    statistics.main()
    done(t)

    step += 1
    t = banner(step, total, "Training and evaluating models")
    from src import train_models
    train_models.main()
    done(t)

    step += 1
    t = banner(step, total, "Complexity sweep")
    from src import complexity_analysis
    complexity_analysis.main()
    done(t)

    step += 1
    if args.skip_robustness:
        print(f"\n[{step}/{total}] Robustness analysis SKIPPED")
    else:
        t = banner(step, total, "Robustness analysis")
        from src import robustness
        robustness.main()
        done(t)

    step += 1
    t = banner(step, total, "Generating figures")
    from src import visualization
    visualization.main()
    done(t)

    fig_dir, res_dir = cfg.output_dirs()
    print(f"\n{'=' * 72}")
    print(f"Pipeline finished in {time.time() - overall:.1f}s")
    print(f"  figures  -> {fig_dir}")
    print(f"  results  -> {res_dir}")
    print(f"  features -> {cfg.FEATURES_CSV}")
    if cfg.is_synthetic():
        print("\nReminder: this run used the synthetic fixture. Not results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
