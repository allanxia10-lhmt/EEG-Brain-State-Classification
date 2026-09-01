"""Shared fixtures.

The suite runs against a tiny synthetic recording built in memory, so it needs
no network and no downloaded data.  That keeps it fast enough to run on every
change, which is the only way a test suite actually gets run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.make_synthetic_fixture import CHANNELS_10_10, simulate_recording  # noqa: E402


@pytest.fixture(scope="session")
def sfreq() -> float:
    return cfg.SFREQ


@pytest.fixture(scope="session")
def raw_pair():
    """One simulated participant, both conditions, as (n_ch, n_times) arrays."""
    return {
        "eyes_open": simulate_recording(1, eyes_closed=False),
        "eyes_closed": simulate_recording(1, eyes_closed=True),
    }


def _make_raw(data, sfreq):
    import mne

    info = mne.create_info(CHANNELS_10_10, sfreq, ch_types="eeg", verbose="error")
    return mne.io.RawArray(data, info, verbose="error")


@pytest.fixture(scope="session")
def epochs_pair(raw_pair, sfreq):
    """Preprocessed epochs for both conditions of one participant."""
    from src import preprocessing as pp

    out = {}
    for cond, data in raw_pair.items():
        raw = pp.preprocess_raw(_make_raw(data, sfreq))
        epochs = pp.make_epochs(raw)
        epochs, _ = pp.reject_epochs(epochs)
        out[cond] = epochs
    return out


@pytest.fixture(scope="session")
def small_feature_table(sfreq):
    """A multi-participant feature table, built entirely in memory."""
    import pandas as pd

    from src import feature_extraction as fe
    from src import preprocessing as pp

    frames = []
    for subject_idx in range(1, 7):
        for cond, closed in (("eyes_open", False), ("eyes_closed", True)):
            raw = pp.preprocess_raw(_make_raw(
                simulate_recording(subject_idx, eyes_closed=closed), sfreq))
            epochs = pp.make_epochs(raw)
            epochs, _ = pp.reject_epochs(epochs)
            df = fe.features_from_epochs(epochs)
            df.insert(0, "epoch_index", np.arange(len(df)))
            df.insert(0, "condition", cond)
            df.insert(0, "subject", f"S{subject_idx:03d}")
            frames.append(df)
    table = pd.concat(frames, ignore_index=True)
    table["label"] = (table["condition"] == cfg.POSITIVE_CLASS).astype(int)
    return table
