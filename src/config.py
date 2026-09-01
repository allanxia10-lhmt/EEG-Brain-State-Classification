"""Central configuration for the EEG brain-state classification project.

Every tunable constant lives here so that no parameter is buried in analysis
code.  Values marked "fixed a priori" were chosen before any classification was
run; changing them invalidates the pre-registration-style claim in the paper.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CHECKSUMS = DATA / "checksums"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"

for _d in (RAW, PROCESSED, CHECKSUMS, FIGURES, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

FEATURES_CSV = PROCESSED / "features.csv"
QC_CSV = PROCESSED / "quality_control.csv"
PROVENANCE_JSON = DATA / "provenance.json"

# ----------------------------------------------------------------- seed
RANDOM_SEED = 42

# ----------------------------------------------------------------- dataset
DATASET_NAME = "EEG Motor Movement/Imagery Dataset (eegmmidb)"
DATASET_VERSION = "1.0.0"
DATASET_DOI = "10.13026/C28G6P"
PHYSIONET_BASE = "https://physionet.org/files/eegmmidb/1.0.0"
PHYSIONET_CHECKSUMS = f"{PHYSIONET_BASE}/SHA256SUMS.txt"

# Optional community mirror, used ONLY when PhysioNet is unreachable.  The
# dataset is ODC-BY, which permits redistribution with attribution, but any
# mirrored file is still checksum-verified against the PhysioNet manifest
# before it is accepted.  Set to None to disable the fallback entirely.
MIRROR_BASE = os.environ.get("EEGMMIDB_MIRROR", None)

N_SUBJECTS = 109
SUBJECTS = [f"S{i:03d}" for i in range(1, N_SUBJECTS + 1)]
RUNS = {"R01": "eyes_open", "R02": "eyes_closed"}
CONDITIONS = ["eyes_open", "eyes_closed"]
POSITIVE_CLASS = "eyes_closed"

SFREQ = 160.0  # nominal; the true rate is read from each EDF header

# Subjects excluded in the published analysis because artifact rejection
# removed an entire condition (a paired design requires both).  Recomputed at
# runtime; this list is the expected outcome, not an input filter.
EXPECTED_EXCLUSIONS = ["S009", "S022", "S060"]

# ----------------------------------------------------------------- bands
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
BAND_NAMES = list(BANDS)

# Alternative definition used in the robustness analysis only.
BANDS_ALT = dict(BANDS, alpha=(7.0, 13.0), theta=(4.0, 7.0))

# ----------------------------------------------------------------- filtering
L_FREQ = 1.0   # fixed a priori
H_FREQ = 45.0  # fixed a priori; below the 60 Hz US mains line, so no notch
FILTER_DESIGN = "firwin"
FILTER_PHASE = "zero"

# ----------------------------------------------------------------- bad channels
BAD_FLAT_UV = 1e-7          # < 0.1 uV std is treated as flat
BAD_LOGVAR_Z = 6.0          # robust-z on log variance
BAD_MIN_CORR = 0.30         # max abs correlation with any other channel

# ----------------------------------------------------------------- epoching
EPOCH_SEC = 2.0             # fixed a priori
EPOCH_OVERLAP = 0.0         # non-overlapping keeps samples closer to independent
WINDOW_SWEEP = [0.5, 1.0, 2.0, 4.0, 8.0]

# ----------------------------------------------------------------- rejection
REJECT_PTP_UV = 250.0       # peak-to-peak, microvolts; fixed a priori
REJECT_CHAN_FRAC = 0.20     # discard epoch if >20% of channels exceed the swing
REJECT_PTP_SWEEP = [None, 200.0, 250.0, 350.0]

# ----------------------------------------------------------------- spectra
WELCH_SEG_SEC = 1.0
WELCH_OVERLAP_SEC = 0.8
PSD_FMIN, PSD_FMAX = 1.0, 45.0
IAF_SEARCH = (7.0, 13.0)    # individual alpha peak search range

# ----------------------------------------------------------------- montage
MONTAGE = "standard_1005"

OCCIPITAL = ["O1", "Oz", "O2"]
OCCIPITAL_REGION = ["P7", "P5", "PO7", "PO3", "O1", "Oz", "Iz", "O2", "PO8"]

TEN_TWENTY = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
]

FRONTAL_HEADBAND = ["Fp1", "Fpz", "Fp2"]

REGIONS = {
    "frontal": ["Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
                "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8"],
    "central": ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
                "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
    "temporal": ["FT7", "FT8", "T7", "T8", "TP7", "TP8"],
    "parietal": ["P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
                 "PO7", "PO3", "POz", "PO4", "PO8"],
    "occipital": ["O1", "Oz", "O2", "Iz"],
}

# Electrode budgets compared in the complexity sweep.  Each entry is
# (label, channel list, n_random_draws_for_comparison).
ELECTRODE_SETS = {
    "all_64": None,                      # None means "every good channel"
    "ten_twenty_19": TEN_TWENTY,
    "occipital_9": OCCIPITAL_REGION,
    "occipital_3": OCCIPITAL,
    "occipital_1": ["Oz"],
    "headband_3": FRONTAL_HEADBAND,
    "headband_1": ["Fpz"],
}
N_RANDOM_SUBSETS = 10

# ----------------------------------------------------------------- modelling
TEST_SIZE = 0.30            # proportion of PARTICIPANTS held out
N_CV_FOLDS = 5
N_BOOTSTRAP = 2000          # cluster bootstrap draws for headline models
LOGREG_C_FIXED = 0.01       # selected on training folds; held fixed in sweeps

PARAM_GRIDS = {
    "logistic_regression": {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]},
    "linear_svm": {"clf__C": [0.001, 0.01, 0.1, 1.0]},
    "random_forest": {"clf__n_estimators": [300],
                      "clf__max_depth": [None, 8, 16],
                      "clf__min_samples_leaf": [1, 4]},
    "hist_gradient_boosting": {"clf__max_iter": [200],
                               "clf__learning_rate": [0.05, 0.1],
                               "clf__max_leaf_nodes": [15, 31]},
}

# ----------------------------------------------------------------- provenance


def read_provenance() -> dict:
    """Return the provenance record written by download_data.py.

    Analysis modules use this to refuse to write into results/ when the data
    are synthetic.  A missing file is treated as unknown provenance.
    """
    if PROVENANCE_JSON.exists():
        return json.loads(PROVENANCE_JSON.read_text())
    return {"source": "unknown", "verified": False, "synthetic": False}


def is_synthetic() -> bool:
    return bool(read_provenance().get("synthetic", False))


def output_dirs() -> tuple[Path, Path]:
    """Return (figures_dir, results_dir), redirected when data are synthetic.

    This is the guard that keeps smoke-test output from ever being mistaken
    for real results.  Synthetic runs write to figures/_smoke and
    results/_smoke, and every figure is additionally watermarked.
    """
    if is_synthetic():
        fig, res = FIGURES / "_smoke", RESULTS / "_smoke"
        fig.mkdir(parents=True, exist_ok=True)
        res.mkdir(parents=True, exist_ok=True)
        return fig, res
    return FIGURES, RESULTS
