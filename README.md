# Diminishing returns in resting-state EEG brain-state classification

[![tests](https://github.com/allanxia10-lhmt/EEG-Brain-State-Classification/actions/workflows/tests.yml/badge.svg)](https://github.com/allanxia10-lhmt/EEG-Brain-State-Classification/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22243161.svg)](https://doi.org/10.5281/zenodo.22243161)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)


Can EEG distinguish eyes-open from eyes-closed brain states, and — the question
that actually matters — **how little does it take?**

The first half has been settled since Berger described the alpha rhythm in
1929. Reporting a high accuracy on this contrast contributes nothing. This
project measures the *marginal value* of each resource instead: electrode
count, recording window length, and model complexity, each varied while the
others are held fixed.

**Headline finding from the published run.** A single occipital electrode with
a single alpha feature reaches ROC-AUC 0.818–0.829 on participants never seen
in training. Sixty-four electrodes, 674 features, and a tuned classifier reach
0.894. Roughly two orders of magnitude more hardware and computation buy about
0.07 AUC. Nonlinear models buy nothing. Electrode *placement* is worth +0.084
AUC at a one-electrode budget and nothing at nine, and recording *time*
substitutes directly for electrodes.

One methodological result is worth as much as the main one: splitting epochs at
random instead of by participant inflates AUC from 0.894 to 0.940 — a single
error that yields more apparent improvement than the entire machine-learning
apparatus does honestly.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Offline smoke test: no network, no download, ~3 minutes.
python run_all.py --synthetic 16

# The real thing: downloads 218 EDF files (~300 MB), several hours.
python run_all.py
```

Or conda: `conda env create -f environment.yml && conda activate eeg-brain-state`.

Requires Python 3.10+.

### Running stages individually

```bash
python src/download_data.py            # fetch and checksum-verify the EDFs
python src/feature_extraction.py       # -> data/processed/features.csv
python src/statistics.py               # paired tests, effect sizes, FDR
python src/train_models.py             # models, LOSO, leakage demonstration
python src/complexity_analysis.py      # the ladder, ablation, window sweep
python src/robustness.py               # 15 preprocessing configurations
python src/visualization.py            # all 12 figures
python src/report.py                   # -> paper/generated_results.md
```

Useful flags: `--subjects N` for a faster real subset, `--skip-download` to
reuse `data/raw/`, `--skip-robustness` to skip the slowest stage. Raw data are
never overwritten; the downloader skips files already on disk.

```bash
pytest tests/ -q        # 42 tests, ~10 seconds, no network needed
```

---

## Synthetic mode, and why it cannot contaminate results

`--synthetic` generates simulated EEG with a deliberately planted occipital
alpha effect, so the pipeline can be exercised without network access. It is
not data, and three independent guards keep it from ever being mistaken for a
result:

1. `download_data.py` stamps `data/provenance.json` with `synthetic: true`.
2. `config.output_dirs()` then redirects everything into `figures/_smoke/` and
   `results/_smoke/`, both gitignored.
3. Every figure is stamped with a red **SYNTHETIC FIXTURE — NOT A RESULT**
   watermark, and `report.py` prints the same warning at the top of its output.

## Data provenance

The analysis uses only the two resting-state baseline runs of each participant
— R01 (eyes open) and R02 (eyes closed) — from the PhysioNet EEG Motor
Movement/Imagery Dataset: 109 adults, 64 channels, 160 Hz, 61 s per run.

`download_data.py` prefers PhysioNet and hashes every retrieved file against
the official SHA-256 manifest. A file that does not match is deleted rather than
used. If PhysioNet is unreachable and `EEGMMIDB_MIRROR` is set, it falls back to
that mirror — but the checksum check is identical either way, so a mirrored file
is only accepted if it is bit-identical to the PhysioNet original.

See `data/README.md` for licence terms and citation requirements.

---

## Method, and the decisions that could bias it

Every preprocessing choice is documented in place in `src/preprocessing.py`,
including its potential for bias. The three that matter most:

- **1–45 Hz band-pass.** The 45 Hz edge sits below the 60 Hz US mains line, so
  no notch filter is needed. The 1 Hz high-pass attenuates lower delta, so
  delta estimates here are conservative. Both edges were fixed before any
  classification was run and applied identically to both conditions, so the
  filter cannot manufacture a group difference.
- **Average reference.** These recordings have no usable physical reference. An
  average reference attenuates spatially broad activity, and in the robustness
  analysis it turns out to be the one preprocessing choice that materially
  changes the outcome (removing it costs 0.044 AUC).
- **Fraction-based epoch rejection.** The conventional max-peak-to-peak rule is
  wrong for a 64-channel montage: the maximum is dominated by the single
  noisiest electrode, and at the usual 150 µV threshold it discarded 97% of
  epochs. The rule used instead discards an epoch only when more than 20% of
  channels exceed 250 µV, which targets global artifacts rather than one bad
  electrode.

ICA is **off** by default. These are 61-second recordings with no EOG or ECG
channel, so ocular components would have to be identified heuristically and a
misidentification removes genuine frontal activity. The robustness analysis
confirms that omitting it changed nothing.

### Leakage control

Each participant contributes ~50 epochs sharing a skull, an electrode
placement, and an alpha peak frequency. A model given epochs from the same
person on both sides of a split can identify the *person* and recall their
condition. Every split here is over participants: `GroupShuffleSplit` for the
held-out set, `GroupKFold` for tuning, scaling fitted inside `Pipeline` folds,
and `assert_no_leakage` raising rather than warning on every split.

`leaky_baseline()` deliberately does it wrong, so the cost of the error can be
measured. It is the only function in the project allowed to be wrong, and it
says so in its docstring.

Confidence intervals use a cluster bootstrap that resamples **participants**.
Resampling epochs would treat ~50 dependent observations as independent and
report intervals several times too narrow.

---

## Repository layout

```
src/config.py                 every tunable constant, in one place
src/download_data.py          retrieval + SHA-256 verification + provenance
src/make_synthetic_fixture.py offline simulated EEG (source-mixing model)
src/preprocessing.py          filtering, bad channels, referencing, epoching
src/feature_extraction.py     Welch PSD -> 674 features
src/statistics.py             paired tests, dz, CIs, BH-FDR
src/train_models.py           models, LOSO, leakage demonstration, bootstrap
src/complexity_analysis.py    ladder, electrode ablation, window sweep
src/robustness.py             15 preprocessing configurations
src/visualization.py          all figures
src/report.py                 regenerates paper tables from results/
run_all.py                    the whole pipeline in one command
tests/                        41 tests, including a permuted-label control
paper/                        manuscript, references, summary, slides
```

## Limitations

Read these before citing anything above.

1. **Condition is confounded with run.** Eyes-open and eyes-closed were
   recorded as separate consecutive runs, so any drift in impedance, alertness,
   or electrode contact between runs is perfectly confounded with condition.
   This is inherent to the paradigm and cannot be fixed within this dataset.
2. **Differential epoch rejection.** More eyes-open epochs are rejected than
   eyes-closed (13.7% vs 6.9%), consistent with blinks. Results are unchanged
   with rejection disabled entirely, but the asymmetry is real.
3. **Short recordings.** 61 s per condition caps the number of long windows;
   the window-length curve had not saturated where the data ran out.
4. **One dataset, one contrast.** These cost curves describe a large, spatially
   focal, low-dimensional effect. A subtler or more distributed contrast —
   workload, attention, early pathology — could easily show a different shape,
   and complexity might well earn its keep there. The claim is that complexity
   is unnecessary *for this problem*, not in general.
5. **Simulated, not measured, deployment.** Electrode subsets were formed by
   discarding channels from a 64-channel average-referenced recording. A real
   one-electrode system has a different reference, different noise, and no
   neighbours to interpolate from, and would likely do somewhat worse.
6. **Homogeneous sample.** Healthy volunteers of unreported demographics, one
   system, one laboratory.

## Citation

If you use this code, please cite the dataset as well:

> Schalk, G.; McFarland, D. J.; Hinterberger, T.; Birbaumer, N.; Wolpaw, J. R.
> BCI2000: A General-Purpose Brain-Computer Interface (BCI) System.
> *IEEE Trans. Biomed. Eng.* **2004**, *51* (6), 1034-1043.

> Goldberger, A. L.; et al. PhysioBank, PhysioToolkit, and PhysioNet.
> *Circulation* **2000**, *101* (23), e215-e220.

Code is MIT licensed. The dataset is not — see `data/README.md`.

---

## Citing this repository

`CITATION.cff` drives GitHub's "Cite this repository" button and seeds Zenodo's
metadata. Fill in the placeholders in that file and in `.zenodo.json` before
publishing anything.

### Making the repo citable

A bare GitHub URL is not a citable artifact. Repositories get renamed, deleted,
and force-pushed, so a link alone cannot support a claim in a paper. What a
reader needs is a frozen snapshot with a persistent identifier:

```bash
# 1. Publish the repository (public, or Zenodo cannot see it)
git init && git add -A
git commit -m "EEG brain-state classification pipeline"
git remote add origin https://github.com/USER/eeg-brain-state-classification
git push -u origin main

# 2. Run the real analysis, then commit its outputs
python run_all.py
python src/report.py
git add results/*.csv figures/*.png data/provenance.json paper/generated_results.md
git commit -m "Results from the run reported in the manuscript"

# 3. Connect Zenodo (zenodo.org -> Log in with GitHub -> flip the repo switch),
#    THEN tag and release. Zenodo only archives releases made after the switch
#    is on, so ordering matters here.
git tag -a v1.0.0 -m "Version reported in the manuscript"
git push origin v1.0.0
# Draft a GitHub Release from that tag. Zenodo mints a DOI within a minute.

# 4. Put the DOI back into CITATION.cff, .zenodo.json, and the manuscript's
#    Code availability statement, then release v1.0.1 with the DOI included.
```

Zenodo issues two DOIs: a **concept DOI** that always resolves to the newest
version, and a **version DOI** pinned to one snapshot. Cite the **version DOI**
in the paper. A reader checking your numbers needs the exact code that produced
them, not whatever `main` has drifted to since.

### Why commit results and figures

`results/*.csv`, `figures/*.png`, and `data/provenance.json` are deliberately
**not** gitignored. Committing them means a reader can verify any number in the
manuscript in seconds rather than re-running a multi-hour analysis on data they
would first have to download. `data/provenance.json` in particular is the
evidence for the checksum-verification claim in the Methods section — without
it, that paragraph is an assertion rather than a record.

Do not commit `data/raw/` (300 MB, and freely available from PhysioNet) or
anything under `_smoke/` (synthetic fixture output, not results).




