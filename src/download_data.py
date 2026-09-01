"""Retrieve the eegmmidb baseline runs and verify their integrity.

Only the two resting-state baseline runs of each participant are needed:
R01 (eyes open) and R02 (eyes closed), 218 files in total.

Retrieval order
---------------
1. PhysioNet (the canonical source).
2. A community mirror, used only if PhysioNet is unreachable and the
   EEGMMIDB_MIRROR environment variable is set.

Either way every file is hashed with SHA-256 and compared against the official
PhysioNet manifest.  A file that does not match is deleted, not used.  The
outcome is recorded in data/provenance.json, which downstream modules read
before they are willing to write into results/.

Usage
-----
    python src/download_data.py                 # real data
    python src/download_data.py --subjects 12   # first 12 participants only
    python src/download_data.py --synthetic 12  # offline smoke-test fixture
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402

TIMEOUT = 30
RETRIES = 3
USER_AGENT = "eeg-brain-state-research/1.0 (academic use)"


# --------------------------------------------------------------------- helpers
def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _get(url: str, timeout: int = TIMEOUT) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def reachable(url: str, timeout: int = 10) -> bool:
    try:
        _get(url, timeout=timeout)
        return True
    except Exception:
        return False


def edf_relpath(subject: str, run: str) -> str:
    return f"{subject}/{subject}{run}.edf"


# ------------------------------------------------------------------- manifest
def fetch_manifest() -> dict[str, str]:
    """Return {relative_path: sha256} from the PhysioNet manifest.

    Cached locally so that a later offline run can still verify files that
    were fetched earlier.
    """
    local = cfg.CHECKSUMS / "SHA256SUMS.txt"
    if not local.exists():
        try:
            local.write_bytes(_get(cfg.PHYSIONET_CHECKSUMS))
            print(f"  manifest downloaded -> {local}")
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"  manifest unavailable ({exc.__class__.__name__}: {exc})")
            return {}
    manifest: dict[str, str] = {}
    for line in local.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 64:
            manifest[parts[-1].lstrip("*./")] = parts[0].lower()
    return manifest


# ------------------------------------------------------------------- download
def download_one(relpath: str, dest: Path, bases: list[str]) -> str | None:
    """Try each base URL in turn.  Returns the base that worked, or None."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for base in bases:
        url = f"{base.rstrip('/')}/{relpath}"
        for attempt in range(1, RETRIES + 1):
            try:
                dest.write_bytes(_get(url))
                return base
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                if attempt == RETRIES:
                    print(f"    {base}: {exc.__class__.__name__} ({exc})")
                else:
                    time.sleep(1.5 * attempt)
            except Exception as exc:  # noqa: BLE001
                print(f"    {base}: {exc}")
                break
    return None


def acquire(subjects: list[str]) -> dict:
    bases = [cfg.PHYSIONET_BASE]
    physionet_up = reachable(f"{cfg.PHYSIONET_BASE}/RECORDS")
    if not physionet_up:
        print("PhysioNet is unreachable from this machine.")
        if cfg.MIRROR_BASE:
            print(f"Falling back to mirror: {cfg.MIRROR_BASE}")
            bases.append(cfg.MIRROR_BASE)
        else:
            print("No mirror configured (set EEGMMIDB_MIRROR to enable one).")
    manifest = fetch_manifest()
    if not manifest:
        print("WARNING: no checksum manifest; files cannot be verified.")

    stats = {"downloaded": 0, "cached": 0, "verified": 0,
             "unverifiable": 0, "mismatch": 0, "failed": 0}
    used_bases: set[str] = set()
    failures: list[str] = []

    total = len(subjects) * len(cfg.RUNS)
    i = 0
    for subject in subjects:
        for run in cfg.RUNS:
            i += 1
            rel = edf_relpath(subject, run)
            dest = cfg.RAW / rel
            if dest.exists() and dest.stat().st_size > 0:
                stats["cached"] += 1
            else:
                print(f"[{i:3d}/{total}] {rel}")
                base = download_one(rel, dest, bases)
                if base is None:
                    stats["failed"] += 1
                    failures.append(rel)
                    dest.unlink(missing_ok=True)
                    continue
                used_bases.add(base)
                stats["downloaded"] += 1

            expected = manifest.get(rel)
            if expected is None:
                stats["unverifiable"] += 1
            elif sha256(dest) == expected:
                stats["verified"] += 1
            else:
                print(f"    CHECKSUM MISMATCH, discarding {rel}")
                dest.unlink(missing_ok=True)
                stats["mismatch"] += 1
                failures.append(rel)

    record = {
        "source": sorted(used_bases) or ["cache"],
        "physionet_reachable": physionet_up,
        "synthetic": False,
        "n_subjects_requested": len(subjects),
        "manifest_entries": len(manifest),
        "stats": stats,
        "failures": failures,
        "verified": stats["mismatch"] == 0 and stats["verified"] > 0,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    cfg.PROVENANCE_JSON.write_text(json.dumps(record, indent=2))
    return record


# ------------------------------------------------------------------ synthetic
def make_synthetic(n_subjects: int) -> dict:
    """Delegate to the fixture generator and stamp provenance as synthetic."""
    from src.make_synthetic_fixture import build_fixture

    build_fixture(n_subjects)
    record = {
        "source": ["synthetic-fixture"],
        "physionet_reachable": False,
        "synthetic": True,
        "n_subjects_requested": n_subjects,
        "verified": False,
        "note": ("Simulated EEG generated by src/make_synthetic_fixture.py. "
                 "For smoke-testing the pipeline only. Any number produced "
                 "from these files is meaningless as a scientific result."),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    cfg.PROVENANCE_JSON.write_text(json.dumps(record, indent=2))
    return record


# ----------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", type=int, default=cfg.N_SUBJECTS,
                    help="number of participants to retrieve (default: all 109)")
    ap.add_argument("--synthetic", type=int, nargs="?", const=12, default=None,
                    metavar="N", help="generate an N-subject offline fixture instead")
    args = ap.parse_args(argv)

    if args.synthetic is not None:
        print(f"Generating synthetic fixture with {args.synthetic} participants.")
        rec = make_synthetic(args.synthetic)
        print("\nSYNTHETIC MODE. Outputs will be written to figures/_smoke "
              "and results/_smoke and are not results.")
    else:
        subjects = cfg.SUBJECTS[: args.subjects]
        print(f"Acquiring {len(subjects) * len(cfg.RUNS)} EDF files "
              f"({len(subjects)} participants x {len(cfg.RUNS)} runs).")
        rec = acquire(subjects)
        s = rec["stats"]
        print("\n--- acquisition summary ---")
        print(f"  downloaded    {s['downloaded']}")
        print(f"  already local {s['cached']}")
        print(f"  verified      {s['verified']}")
        print(f"  unverifiable  {s['unverifiable']}")
        print(f"  mismatched    {s['mismatch']}")
        print(f"  failed        {s['failed']}")
        if s["failed"] or s["mismatch"]:
            print("\nSome files are missing. Re-run to retry; the pipeline "
                  "will use whatever verified files are present.")
            return 1
    print(f"\nprovenance -> {cfg.PROVENANCE_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
