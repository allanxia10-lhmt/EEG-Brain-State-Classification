"""Preflight check before archiving this repository under a DOI.

A published Zenodo record cannot be deleted.  Whatever you archive is what a
reader will find in twenty years, so this script refuses to bless a release
that is not actually ready.

    python src/check_release.py

Exits 0 if every check passes, 1 otherwise.  Nothing here talks to Zenodo or
GitHub; it only inspects the repository you are about to publish.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402

ROOT = cfg.ROOT
PLACEHOLDERS = [r"\[REPLACE\]", r"REPLACE:", r"REPLACE-USER", r"\[Zenodo DOI\]",
                r"\[repository URL\]", r"\[School Name\]", r"\[Street Address\]",
                r"\[author email\]", r"\[City\]", r"\[State\]", r"\[Postal Code\]"]

results: list[tuple[bool, str, str]] = []


def check(ok: bool, title: str, detail: str = "") -> None:
    results.append((ok, title, detail))


# --------------------------------------------------------------- provenance
def check_provenance() -> None:
    prov = cfg.read_provenance()
    if prov.get("source") == "unknown":
        check(False, "Data provenance recorded",
              "data/provenance.json is missing. Run the real pipeline first.")
        return
    if prov.get("synthetic"):
        check(False, "Analysis used real data",
              "The last run used the SYNTHETIC fixture. Archiving this would "
              "publish simulated numbers as if they were results.")
        return
    check(True, "Analysis used real data", f"source: {prov.get('source')}")
    check(bool(prov.get("verified")), "Dataset checksums verified",
          "" if prov.get("verified") else
          "provenance.json reports unverified files. The Methods section "
          "claims checksum verification; that claim must be true.")


# ------------------------------------------------------------- placeholders
def check_placeholders() -> None:
    patterns = re.compile("|".join(PLACEHOLDERS))
    targets = [ROOT / "CITATION.cff", ROOT / ".zenodo.json",
               ROOT / "README.md", ROOT / "paper" / "research_paper.md",
               ROOT / "paper" / "references.md",
               ROOT / "paper" / "research_summary.md"]
    dirty = []
    for path in targets:
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if patterns.search(line) and not line.lstrip().startswith("#"):
                dirty.append(f"{path.relative_to(ROOT)}:{i}")
    check(not dirty, "No unfilled placeholders",
          f"{len(dirty)} found: " + ", ".join(dirty[:6]) +
          ("..." if len(dirty) > 6 else "") if dirty else "")


# ----------------------------------------------------------------- metadata
def check_metadata() -> None:
    cff = ROOT / "CITATION.cff"
    zen = ROOT / ".zenodo.json"
    check(cff.exists(), "CITATION.cff present",
          "" if cff.exists() else "GitHub and Zenodo both read this for metadata.")
    if zen.exists():
        try:
            json.loads(zen.read_text())
            check(True, ".zenodo.json is valid JSON")
        except json.JSONDecodeError as exc:
            check(False, ".zenodo.json is valid JSON", str(exc))
    else:
        check(False, ".zenodo.json present", "Zenodo will fall back to guessed metadata.")

    if cff.exists():
        text = cff.read_text()
        # Must be the TOP-LEVEL doi key. An indented one belongs to an entry
        # in the references block -- the dataset's DOI is not the software's,
        # and matching it here produced a false pass.
        has_doi = re.search(r"^doi:\s*['\"]?10\.", text, re.M)
        check(bool(has_doi), "CITATION.cff carries a DOI",
              "" if has_doi else
              "Expected on the SECOND release. On a first manual upload, "
              "reserve the DOI in Zenodo and paste it in before uploading.")


# ------------------------------------------------------------------ outputs
def check_outputs() -> None:
    _, res = cfg.output_dirs()
    csvs = sorted(p for p in res.glob("*.csv") if p.stat().st_size > 0)
    figs = sorted(p for p in cfg.FIGURES.glob("*.png"))
    check(len(csvs) >= 8, "Results committed",
          f"only {len(csvs)} non-empty CSVs in results/. A reader needs these "
          "to check the paper without re-running for hours.")
    check(len(figs) >= 9, "Figures committed", f"only {len(figs)} PNGs in figures/.")

    gen = cfg.PAPER / "generated_results.md"
    if not gen.exists():
        check(False, "generated_results.md present",
              "Run `python src/report.py`. This is the file you diff the "
              "manuscript against.")
        return
    stale = [p.name for p in csvs if p.stat().st_mtime > gen.stat().st_mtime]
    check(not stale, "generated_results.md is current",
          f"older than {len(stale)} results file(s): {', '.join(stale[:4])}. "
          "Re-run `python src/report.py`." if stale else "")


# -------------------------------------------------------------------- tests
def check_tests() -> None:
    try:
        proc = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                              cwd=ROOT, capture_output=True, text=True, timeout=900)
        tail = (proc.stdout.strip().splitlines() or [""])[-1]
        check(proc.returncode == 0, "Test suite passes", tail)
    except Exception as exc:  # noqa: BLE001
        check(False, "Test suite passes", f"could not run pytest: {exc}")


# ---------------------------------------------------------------------- git
def check_git() -> None:
    def git(*args) -> tuple[int, str]:
        p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return p.returncode, p.stdout.strip()

    code, _ = git("rev-parse", "--git-dir")
    if code != 0:
        check(False, "Git repository initialised",
              "Not a git repo yet. Zenodo archives GitHub releases.")
        return
    check(True, "Git repository initialised")

    _, dirty = git("status", "--porcelain")
    lines = [l for l in dirty.splitlines() if l.strip()]
    check(not lines, "Working tree clean",
          f"{len(lines)} uncommitted change(s). The archive is built from a "
          "tag, so anything uncommitted will not be in it." if lines else "")

    code, remotes = git("remote", "-v")
    check(bool(remotes), "Remote configured",
          "" if remotes else "No git remote. Zenodo can only see a pushed repo.")


def main() -> int:
    print("Release preflight\n" + "=" * 66)
    check_provenance()
    check_placeholders()
    check_metadata()
    check_outputs()
    check_tests()
    check_git()

    failures = 0
    for ok, title, detail in results:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {title}")
        if detail:
            for line in _wrap(detail):
                print(f"         {line}")
        failures += (not ok)

    print("=" * 66)
    if failures:
        print(f"{failures} of {len(results)} checks failed.\n\n"
              "Do NOT publish yet. A Zenodo record cannot be deleted once\n"
              "published, so fix these first. To rehearse the whole process\n"
              "safely, use sandbox.zenodo.org, which issues throwaway DOIs.")
        return 1
    print(f"All {len(results)} checks passed. Safe to tag and archive.\n"
          "Reminder: cite the VERSION DOI in the manuscript, not the concept DOI.")
    return 0


def _wrap(text: str, width: int = 62) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
