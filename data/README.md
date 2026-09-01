# Data

Nothing in this directory is committed to version control. The raw EEG is about
300 MB for the baseline runs alone, and it is freely redistributable from its
canonical home, so mirroring it in git would be pointless as well as large.

Run `python src/download_data.py` to populate `raw/`.

## Dataset

| Field | Value |
|---|---|
| Name | EEG Motor Movement/Imagery Dataset (eegmmidb), v1.0.0 |
| Source | PhysioNet |
| URL | https://physionet.org/content/eegmmidb/1.0.0/ |
| DOI | 10.13026/C28G6P |
| Participants | 109 adults |
| Channels | 64, extended 10-10 montage |
| Sampling rate | 160 Hz |
| Acquisition | BCI2000 |
| Licence | Open Data Commons Attribution License v1.0 (ODC-BY) |

### Runs used

The full dataset contains 14 runs per participant, mostly motor movement and
motor imagery tasks. This project uses **only the two resting-state baselines**:

| Run | Condition | Duration | Label |
|---|---|---|---|
| R01 | Eyes open | 61 s | 0 |
| R02 | Eyes closed | 61 s | 1 |

Run identity supplies the condition label, so no annotation parsing is needed
and there is no task confound. Each participant contributes exactly one
recording per condition, which makes the design fully paired within-participant.

### Why this dataset

- 109 participants is enough for participant-level cross-validation to mean
  something. Most publicly available resting-state EEG has fewer than 30.
- The two baseline runs give a clean binary contrast with no task overlay.
- The licence permits redistribution and independent verification, so the
  analysis can actually be checked by someone else.
- 64 channels allow the electrode-ablation analysis to subset downward, which a
  low-density dataset could not support.

### Licence terms

ODC-BY permits copying, distribution, and adaptation, including commercially,
provided attribution is given. Attribution requirements are satisfied by citing
both the BCI2000 paper and PhysioNet — see the citation block in the top-level
README.

**This licence covers the data only.** The code in this repository is MIT
licensed; the two are separate.

## Layout

```
data/
├── raw/                  S001/S001R01.edf ... S109/S109R02.edf   (218 files)
├── processed/
│   ├── features.csv      one row per epoch, 674 features + metadata
│   └── quality_control.csv   per-recording bad channels and rejection rates
├── checksums/
│   └── SHA256SUMS.txt    cached PhysioNet manifest
└── provenance.json       where the data came from and whether it verified
```

## Integrity

Every downloaded file is hashed with SHA-256 and compared against the official
PhysioNet manifest. A mismatched file is deleted, not used. The outcome is
recorded in `provenance.json`, and the analysis modules read that record before
they are willing to write into `results/` — a run on synthetic data is
redirected to `results/_smoke/` automatically.

If PhysioNet is unreachable and the `EEGMMIDB_MIRROR` environment variable is
set, the downloader falls back to that mirror. The checksum check is identical
either way, so a mirrored file is accepted only if it is bit-identical to the
PhysioNet original. Note that the manifest itself comes from PhysioNet, so a
fully offline machine cannot verify files it has never seen before.

## Excluded participants

Three participants (S009, S022, S060) lost an entire condition to artifact
rejection in the published run and were excluded, because a paired design
requires both conditions from the same person. This is recomputed at runtime
rather than hard-coded as a filter — if your rejection settings differ, your
exclusions will differ, and `quality_control.csv` records what actually
happened.
