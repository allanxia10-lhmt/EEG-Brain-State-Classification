"""Configuration, acquisition, and preprocessing tests."""
from __future__ import annotations

import numpy as np
import pytest

from src import config as cfg
from src import download_data as dl
from src import preprocessing as pp


# ------------------------------------------------------------------- config
def test_bands_are_contiguous_and_ordered():
    edges = list(cfg.BANDS.values())
    for (lo, hi) in edges:
        assert lo < hi
    for (_, hi), (lo2, _) in zip(edges, edges[1:]):
        assert hi == lo2, "band edges must tile the spectrum without gaps"


def test_required_channels_exist_in_montage():
    """Every channel the analysis names must be placeable, or it is dropped."""
    import mne

    montage = set(mne.channels.make_standard_montage(cfg.MONTAGE).ch_names)
    named = set(cfg.OCCIPITAL) | set(cfg.TEN_TWENTY) | set(cfg.FRONTAL_HEADBAND)
    named |= set(cfg.OCCIPITAL_REGION)
    for members in cfg.REGIONS.values():
        named |= set(members)
    missing = sorted(named - montage)
    assert not missing, f"channels absent from {cfg.MONTAGE}: {missing}"


def test_output_dirs_redirect_under_synthetic_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "FIGURES", tmp_path / "figures")
    monkeypatch.setattr(cfg, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(cfg, "is_synthetic", lambda: True)
    fig, res = cfg.output_dirs()
    assert fig.name == "_smoke" and res.name == "_smoke"


# ------------------------------------------------------------------ download
def test_sha256_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "blob.bin"
    payload = b"eeg" * 5000
    path.write_bytes(payload)
    assert dl.sha256(path) == hashlib.sha256(payload).hexdigest()


def test_edf_relpath_layout():
    assert dl.edf_relpath("S001", "R02") == "S001/S001R02.edf"


# --------------------------------------------------------------- preprocessing
def test_channel_names_resolve_to_montage_spelling(raw_pair, sfreq):
    """Regression test for a bug that silently deleted Fp1, Fpz and Fp2.

    A prefix-based renaming rule produced FP1/FPz/FP2, which standard_1005
    spells Fp1/Fpz/Fp2.  The montage then dropped all three without warning,
    removing exactly the electrodes the frontal-headband comparison needs.
    """
    import mne

    from tests.conftest import _make_raw

    raw = _make_raw(raw_pair["eyes_open"], sfreq)
    pp.clean_channel_names(raw)
    pp.attach_montage(raw)
    for ch in ("Fp1", "Fpz", "Fp2", "Oz", "Iz", "FCz", "CPz", "AFz", "POz"):
        assert ch in raw.ch_names, f"{ch} was lost during renaming"
    assert len(raw.ch_names) == 64


def test_preprocessing_preserves_all_64_channels(raw_pair, sfreq):
    from tests.conftest import _make_raw

    raw = pp.preprocess_raw(_make_raw(raw_pair["eyes_closed"], sfreq))
    assert len(raw.ch_names) == 64


def test_bandpass_attenuates_out_of_band_power(raw_pair, sfreq):
    """Energy above the low-pass edge should be strongly reduced."""
    from scipy.signal import welch

    from tests.conftest import _make_raw

    raw = pp.preprocess_raw(_make_raw(raw_pair["eyes_open"], sfreq))
    freqs, psd = welch(raw.get_data(), fs=sfreq, nperseg=int(sfreq))
    inband = psd[:, (freqs >= 5) & (freqs <= 40)].mean()
    above = psd[:, freqs > cfg.H_FREQ + 10].mean()
    assert above < inband * 0.05


def test_average_reference_zeroes_the_spatial_mean(raw_pair, sfreq):
    from tests.conftest import _make_raw

    raw = pp.preprocess_raw(_make_raw(raw_pair["eyes_open"], sfreq),
                            average_reference=True)
    assert np.abs(raw.get_data().mean(axis=0)).max() < 1e-12


def test_bad_channel_detection_finds_a_flat_channel(raw_pair, sfreq):
    from tests.conftest import _make_raw

    data = raw_pair["eyes_open"].copy()
    data[7] = 0.0
    raw = _make_raw(data, sfreq)
    pp.clean_channel_names(raw)
    pp.attach_montage(raw)
    assert raw.ch_names[7] in pp.detect_bad_channels(raw)


def test_bad_channel_detection_spares_noisy_but_working_frontals(raw_pair, sfreq):
    """Frontal channels carry large ocular variance and are not broken.

    An early variance-only rule flagged every frontal electrode on the first
    participant.  Correlation with neighbours is what separates "noisy" from
    "dead", and this test pins that behaviour down.
    """
    from tests.conftest import _make_raw

    raw = _make_raw(raw_pair["eyes_open"], sfreq)
    pp.clean_channel_names(raw)
    pp.attach_montage(raw)
    raw.filter(cfg.L_FREQ, cfg.H_FREQ, verbose="error")
    bads = set(pp.detect_bad_channels(raw))
    frontal = {"Fp1", "Fpz", "Fp2"}
    assert not frontal <= bads, "all frontal channels flagged; rule is too eager"


def test_epochs_are_non_overlapping_and_correctly_sized(raw_pair, sfreq):
    from tests.conftest import _make_raw

    raw = pp.preprocess_raw(_make_raw(raw_pair["eyes_open"], sfreq))
    epochs = pp.make_epochs(raw, duration=2.0)
    assert epochs.get_data(copy=False).shape[2] == int(2.0 * sfreq)
    starts = epochs.events[:, 0] / sfreq
    assert np.all(np.diff(starts) >= 2.0 - 1e-6)


def test_rejection_removes_epochs_with_global_artifacts(epochs_pair):
    import mne

    epochs = epochs_pair["eyes_open"].copy()
    data = epochs.get_data().copy()
    # A DC offset would not change peak-to-peak at all, so the artifact has to
    # be a swing: a large ramp across every channel of one epoch.
    ramp = np.linspace(-5e-4, 5e-4, data.shape[2])
    data[0, :, :] += ramp[None, :]
    spiked = mne.EpochsArray(data, epochs.info, verbose="error")
    kept, mask = pp.reject_epochs(spiked, ptp_uv=250.0)
    assert not mask[0]
    assert len(kept) == len(spiked) - 1


def test_rejection_ignores_a_single_noisy_channel(epochs_pair):
    """The rule targets global artifacts, not one bad electrode.

    A max-across-channels rule would discard this epoch.  At 64 channels that
    behaviour rejected 97% of the dataset, which is why the fraction-based rule
    exists.
    """
    import mne

    epochs = epochs_pair["eyes_open"].copy()
    data = epochs.get_data().copy()
    data[0, 3, :] += 1e-3          # one channel, one epoch
    spiked = mne.EpochsArray(data, epochs.info, verbose="error")
    _, mask = pp.reject_epochs(spiked, ptp_uv=250.0, chan_frac=0.20)
    assert mask[0], "a single noisy channel should not discard the epoch"


def test_disabling_rejection_keeps_every_epoch(epochs_pair):
    epochs = epochs_pair["eyes_closed"]
    kept, mask = pp.reject_epochs(epochs, ptp_uv=1e9)
    assert mask.all() and len(kept) == len(epochs)


def test_preprocessing_never_sees_the_condition_label(raw_pair, sfreq):
    """Identical input must give identical output regardless of condition.

    Preprocessing that behaved differently per condition could manufacture a
    group difference out of nothing.
    """
    from tests.conftest import _make_raw

    data = raw_pair["eyes_open"]
    a = pp.preprocess_raw(_make_raw(data.copy(), sfreq)).get_data()
    b = pp.preprocess_raw(_make_raw(data.copy(), sfreq)).get_data()
    assert np.allclose(a, b)
