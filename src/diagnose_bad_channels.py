"""Diagnose the bad-channel detector: clean data, or a rule that never fires?

The main analysis flagged no channels at all across 216 recordings.  That has
two very different explanations, and reporting either one without testing would
be a guess:

  A. The dataset really is clean and no electrode is faulty.
  B. The thresholds are so permissive that the rule could not fire on anything
     short of a disconnected wire.

Three tests separate them.

  1. MARGIN  -- how close did the nearest channel actually come to each
     threshold?  If the worst channel in the whole dataset sits at robust-z 3.2
     against a threshold of 6, nothing was near the line and A is supported.
     If channels pile up at 5.8, the threshold is doing arbitrary work.

  2. SWEEP   -- vary each threshold and count flags.  A cliff (0 flags at the
     chosen value, hundreds just past it) means the setting is load-bearing and
     poorly placed.  A gentle slope means there is no hidden population of bad
     channels waiting to be found.

  3. INJECT  -- deliberately corrupt a channel in a real recording and check
     that the detector catches it.  This is the decisive test.  If a planted
     dead electrode is NOT caught, the rule is broken and zero means nothing.
     If it is caught, the rule works at this dataset's real noise level and
     zero is a finding rather than an artifact.

Usage
-----
    python src/diagnose_bad_channels.py                # 20 recordings
    python src/diagnose_bad_channels.py --n 108        # everything
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402
from src import preprocessing as pp  # noqa: E402

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ statistics
def channel_stats(raw) -> dict[str, np.ndarray]:
    """The three quantities the detector thresholds, per channel."""
    data = raw.get_data()
    sd = data.std(axis=1)

    live = sd >= cfg.BAD_FLAT_UV
    robust_z = np.zeros_like(sd)
    max_corr = np.ones_like(sd)

    if live.sum() >= 3:
        logvar = np.log(data[live].var(axis=1) + 1e-30)
        med = np.median(logvar)
        mad = np.median(np.abs(logvar - med)) + 1e-12
        robust_z[live] = 0.6745 * (logvar - med) / mad

        corr = np.corrcoef(data[live])
        np.fill_diagonal(corr, np.nan)
        max_corr[live] = np.nanmax(np.abs(corr), axis=1)

    return {"sd": sd, "abs_robust_z": np.abs(robust_z), "max_corr": max_corr}


def load_prepared(subject: str, run: str):
    """Filtered and montaged, but BEFORE bad-channel detection."""
    import mne

    path = cfg.RAW / subject / f"{subject}{run}.edf"
    if not path.exists():
        return None
    raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
    pp.clean_channel_names(raw)
    pp.attach_montage(raw)
    raw.filter(cfg.L_FREQ, cfg.H_FREQ, method="fir",
               fir_design=cfg.FILTER_DESIGN, phase=cfg.FILTER_PHASE,
               verbose="error")
    return raw


# ---------------------------------------------------------------- 1. margins
def test_margin(raws) -> dict:
    print("\n1. MARGIN -- how close did anything come to a threshold?\n")
    z_all, corr_all, sd_all = [], [], []
    for raw in raws:
        st = channel_stats(raw)
        z_all.append(st["abs_robust_z"])
        corr_all.append(st["max_corr"])
        sd_all.append(st["sd"])
    z = np.concatenate(z_all); c = np.concatenate(corr_all); s = np.concatenate(sd_all)

    print(f"   channels examined: {len(z)}")
    print(f"   {'':22s} {'threshold':>10s} {'worst seen':>12s} {'margin':>10s}")
    print(f"   {'log-variance |z|':22s} {cfg.BAD_LOGVAR_Z:>10.2f} {z.max():>12.2f} "
          f"{cfg.BAD_LOGVAR_Z - z.max():>10.2f}")
    print(f"   {'max correlation':22s} {cfg.BAD_MIN_CORR:>10.2f} {c.min():>12.2f} "
          f"{c.min() - cfg.BAD_MIN_CORR:>10.2f}")
    print(f"   {'channel SD (uV)':22s} {cfg.BAD_FLAT_UV*1e6:>10.3f} "
          f"{s.min()*1e6:>12.3f} {(s.min()-cfg.BAD_FLAT_UV)*1e6:>10.3f}")

    print(f"\n   |z| percentiles   50th {np.percentile(z,50):.2f}  "
          f"90th {np.percentile(z,90):.2f}  99th {np.percentile(z,99):.2f}  "
          f"max {z.max():.2f}")
    print(f"   corr percentiles  1st {np.percentile(c,1):.2f}  "
          f"10th {np.percentile(c,10):.2f}  50th {np.percentile(c,50):.2f}  "
          f"min {c.min():.2f}")

    verdict = ("data are genuinely clean; nothing approached the line"
               if z.max() < cfg.BAD_LOGVAR_Z * 0.6 and c.min() > cfg.BAD_MIN_CORR + 0.25
               else "some channels came close; the threshold placement matters")
    print(f"\n   -> {verdict}")
    return {"z": z, "corr": c, "sd": s}


# ------------------------------------------------------------------ 2. sweep
def test_sweep(stats: dict) -> None:
    print("\n2. SWEEP -- is the chosen threshold sitting on a cliff?\n")
    z, c = stats["z"], stats["corr"]
    n = len(z)

    print("   log-variance robust-z")
    for thr in (2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
        k = int((z > thr).sum())
        mark = "  <- current" if thr == cfg.BAD_LOGVAR_Z else ""
        print(f"     |z| > {thr:<4.1f}  {k:5d} channels ({100*k/n:5.2f}%){mark}")

    print("\n   maximum absolute correlation")
    for thr in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
        k = int((c < thr).sum())
        mark = "  <- current" if abs(thr - cfg.BAD_MIN_CORR) < 1e-9 else ""
        extra = "  (PREP default)" if abs(thr - 0.40) < 1e-9 else ""
        print(f"     corr < {thr:<4.2f} {k:5d} channels ({100*k/n:5.2f}%){mark}{extra}")


# ----------------------------------------------------------------- 3. inject
def test_injection(raws, labels) -> None:
    """Plant a known-bad channel in a real recording and see if it is caught.

    This is the test that actually settles the question.  A rule that cannot
    detect a channel you broke on purpose tells you nothing when it reports
    zero.
    """
    import mne

    print("\n3. INJECT -- can the rule catch a channel broken on purpose?\n")

    def corrupt(data, idx, mode, rng):
        d = data.copy()
        if mode == "dead (flat)":
            d[idx] = 0.0
        elif mode == "dead + tiny noise":
            d[idx] = rng.standard_normal(d.shape[1]) * 1e-9
        elif mode == "10x amplitude":
            d[idx] = d[idx] * 10.0
        elif mode == "decorrelated noise":
            d[idx] = rng.standard_normal(d.shape[1]) * d[idx].std()
        elif mode == "railing (clipped)":
            lim = np.percentile(np.abs(d[idx]), 20)
            d[idx] = np.clip(d[idx] * 50, -lim, lim)
        return d

    modes = ["dead (flat)", "dead + tiny noise", "10x amplitude",
             "decorrelated noise", "railing (clipped)"]
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    caught = {m: 0 for m in modes}
    total = 0

    for raw, label in zip(raws, labels):
        data = raw.get_data()
        info = raw.info
        for mode in modes:
            idx = int(rng.integers(0, data.shape[0]))
            broken = mne.io.RawArray(corrupt(data, idx, mode, rng), info,
                                     verbose="error")
            broken.info["bads"] = []
            flagged = pp.detect_bad_channels(broken)
            if raw.ch_names[idx] in flagged:
                caught[mode] += 1
        total += 1

    print(f"   {total} recordings, one channel corrupted per mode\n")
    print(f"   {'corruption':24s} {'caught':>8s} {'rate':>8s}")
    for mode in modes:
        rate = caught[mode] / total if total else 0
        flag = "" if rate >= 0.9 else ("   <- MISSED" if rate < 0.5 else "   <- partial")
        print(f"   {mode:24s} {caught[mode]:>4d}/{total:<3d} {rate:>7.0%}{flag}")

    hard_fail = caught["dead (flat)"] < total or caught["dead + tiny noise"] < total
    print()
    if hard_fail:
        print("   -> The rule misses a channel that is plainly dead. Zero flags")
        print("      in the main analysis means nothing. Fix the detector.")
    elif all(caught[m] / total >= 0.8 for m in modes):
        print("   -> The rule catches every corruption type at this dataset's")
        print("      real noise level. Zero flags is a finding, not an artifact.")
    else:
        missed = [m for m in modes if caught[m] / total < 0.8]
        print("   -> Obvious faults are caught, but these are not:")
        for m in missed:
            print(f"        {m}")
        print("      Zero flags means no gross faults, not no bad channels.")


# ------------------------------------------------------------------- driver
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20,
                    help="number of participants to examine (default 20)")
    args = ap.parse_args()

    subjects = pp.available_subjects()[: args.n]
    if not subjects:
        print("No recordings in data/raw. Run src/download_data.py first.")
        return 1

    print("=" * 68)
    print("Bad-channel detector diagnostic")
    print("=" * 68)
    print(f"thresholds: |robust-z| > {cfg.BAD_LOGVAR_Z}, "
          f"max corr < {cfg.BAD_MIN_CORR}, SD < {cfg.BAD_FLAT_UV*1e6:.3f} uV")
    print(f"loading {len(subjects)} participants (eyes-open run)...")

    raws, labels = [], []
    for i, subject in enumerate(subjects, 1):
        raw = load_prepared(subject, "R01")
        if raw is not None:
            raws.append(raw); labels.append(subject)
        if i % 5 == 0:
            print(f"  {i}/{len(subjects)}")
    print(f"loaded {len(raws)} recordings, {len(raws[0].ch_names)} channels each")

    stats = test_margin(raws)
    test_sweep(stats)
    test_injection(raws, labels)

    print("\n" + "=" * 68)
    print("Read tests 1 and 3 together. Test 3 is decisive: if planted faults")
    print("are caught, the rule works and zero is real. Test 1 then tells you")
    print("whether zero was comfortable or a near miss.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
