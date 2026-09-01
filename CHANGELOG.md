# Changelog

Versions follow [semantic versioning](https://semver.org). Each tagged release
is archived to Zenodo with its own DOI; cite the **version** DOI matching the
code that produced the numbers you are quoting.

## [Unreleased]

Not yet released. `python src/check_release.py` currently fails, and the
outstanding items are listed below.

### Blocking a v1.0.0 tag

- The analysis has not been re-run on real data since the channel-naming fix
  (see below). Until it is, the manuscript's numbers are unverified against
  this code.
- Placeholders remain unfilled: Zenodo DOI, repository URL, author email, city
  and state.
- `results/*.csv`, `figures/*.png`, and `data/provenance.json` are not yet
  committed, so a reader cannot check the paper without a multi-hour rerun.

### Fixed

- **Channel names were resolved by a prefix heuristic that produced `FP1`,
  `FPz` and `FP2`.** The `standard_1005` montage spells these `Fp1`, `Fpz`,
  `Fp2`, so all three were dropped without any warning — precisely the
  electrodes the frontal-headband comparison depends on. Names are now
  canonicalised against the montage itself, and a regression test pins it.
  **Any result computed before this fix should be recomputed.**
- **The synthetic fixture wrote EDF files clipped to ±0.01 µV**, because MNE's
  EDF exporter reads `physical_range` in microvolts rather than volts. Every
  band power collapsed to zero and every model reported ROC-AUC exactly 0.500 —
  a null result that looked entirely plausible and crashed nothing.
- **The synthetic fixture used independent per-channel noise**, which is not how
  EEG behaves; real channels correlate because they share sources. The
  correlation-based bad-channel rule therefore flagged all 64 as broken. The
  fixture is now a source-mixing model using real montage positions.
- `fillna` was passed a NumPy array, which pandas 3.0 rejects.
- `check_release.py` matched the *dataset's* DOI in the references block and
  reported a software DOI as present when none was.

### Added

- `_assert_features_are_live()`, which raises when the feature matrix is
  degenerate. A silently zeroed matrix is this pipeline's worst failure mode
  because it does not crash — a null result must come from the data, never from
  a unit bug.
- ROC curve figure (`09_roc_curve.png`), omitted from the original figure set.
- `src/report.py`, which regenerates every numeric claim from `results/*.csv`
  into `paper/generated_results.md`, so the manuscript can be diffed against an
  actual run.
- `src/check_release.py`, a ten-point preflight check that refuses to bless a
  release while the repository is not publication-ready.
- `CITATION.cff` and `.zenodo.json` for citation and archival metadata.
- Continuous integration: unit tests on Python 3.11 and 3.12, plus an
  end-to-end pipeline run that asserts the synthetic-data quarantine held.

### Changed

- Test suite grown to 41 tests, including a permuted-label negative control and
  regression tests for both bugs above.
- Affiliation recorded as independent research, with no institutional funding
  or supervision.
