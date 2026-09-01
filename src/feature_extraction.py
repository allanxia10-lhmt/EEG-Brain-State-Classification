"""Spectral feature extraction.

Each 2-second epoch becomes one row of 674 features:

    logabs_<ch>_<band>     320   log10 absolute band power, uV^2
    rel_<ch>_<band>        320   band power / total 1-45 Hz power
    region_<region>_<band>  25   mean relative power over a scalp region
    global_<band>            5   mean relative power across all channels
    derived                  4   two occipital ratios, an asymmetry, and IAF
    ----------------------------
                           674

Why both absolute and relative power?  Relative power is insensitive to
overall gain -- skull thickness, electrode impedance, amplifier settings --
which makes it comparable across participants.  But it is also compositional:
the five bands sum to one, so a genuine rise in one band mechanically depresses
every other.  Keeping absolute power alongside it is what lets the statistics
module tell a real change from a renormalisation artifact.

Nothing here is fitted across epochs, so feature extraction cannot leak
information between participants.  Standardisation is deliberately deferred to
the modelling stage, where it is fitted inside each cross-validation fold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402
from src import preprocessing as pp  # noqa: E402


# ------------------------------------------------------------------ spectra
def epoch_psd(data: np.ndarray, sfreq: float,
              seg_sec: float | None = None,
              overlap_sec: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density for every epoch and channel.

    ``data`` is (n_epochs, n_channels, n_times) in volts.  Returns
    ``(freqs, psd)`` with ``psd`` shaped (n_epochs, n_channels, n_freqs) in
    uV^2/Hz.

    Welch's method averages periodograms over overlapping sub-windows, trading
    frequency resolution for a lower-variance estimate.  With 1-second segments
    at 0.8-second overlap a 2-second epoch yields six averaged segments at 1 Hz
    resolution, which is comfortably finer than the narrowest band edge.
    """
    seg_sec = cfg.WELCH_SEG_SEC if seg_sec is None else seg_sec
    overlap_sec = cfg.WELCH_OVERLAP_SEC if overlap_sec is None else overlap_sec

    nperseg = int(round(min(seg_sec, data.shape[-1] / sfreq) * sfreq))
    nperseg = max(16, min(nperseg, data.shape[-1]))
    noverlap = int(round(min(overlap_sec, seg_sec * 0.9) * sfreq))
    noverlap = max(0, min(noverlap, nperseg - 1))

    freqs, psd = welch(data * 1e6, fs=sfreq, window="hann", nperseg=nperseg,
                       noverlap=noverlap, axis=-1, scaling="density")
    keep = (freqs >= cfg.PSD_FMIN) & (freqs <= cfg.PSD_FMAX)
    return freqs[keep], psd[..., keep]


def band_power(freqs: np.ndarray, psd: np.ndarray,
               bands: dict | None = None) -> dict[str, np.ndarray]:
    """Integrate the PSD over each band.  Returns {band: (n_epochs, n_ch)}."""
    bands = cfg.BANDS if bands is None else bands
    out = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        out[name] = (np.trapezoid(psd[..., mask], freqs[mask], axis=-1)
                     if mask.sum() > 1 else np.zeros(psd.shape[:-1]))
    return out


def individual_alpha_peak(freqs: np.ndarray, psd: np.ndarray,
                          picks: list[int]) -> np.ndarray:
    """Frequency of the spectral maximum within 7-13 Hz at occipital channels.

    Alpha peak frequency varies systematically between individuals, so a fixed
    8-13 Hz window is not equally centred on everyone's rhythm.  Retaining the
    peak as its own feature lets a model use that variation instead of being
    hurt by it.
    """
    mask = (freqs >= cfg.IAF_SEARCH[0]) & (freqs <= cfg.IAF_SEARCH[1])
    if mask.sum() == 0 or not picks:
        return np.full(psd.shape[0], np.nan)
    occ = psd[:, picks, :][..., mask].mean(axis=1)
    return freqs[mask][np.argmax(occ, axis=1)]


# ------------------------------------------------------------------ features
def features_from_epochs(epochs, bands: dict | None = None,
                         seg_sec: float | None = None,
                         overlap_sec: float | None = None) -> pd.DataFrame:
    """Turn one participant-condition's epochs into a feature table."""
    bands = cfg.BANDS if bands is None else bands
    names = list(epochs.ch_names)
    sfreq = float(epochs.info["sfreq"])
    data = epochs.get_data(copy=False)

    freqs, psd = epoch_psd(data, sfreq, seg_sec, overlap_sec)
    bp = band_power(freqs, psd, bands)

    total = np.sum([bp[b] for b in bands], axis=0) + 1e-30   # (n_ep, n_ch)
    cols: dict[str, np.ndarray] = {}

    for band in bands:
        absolute = bp[band]
        relative = absolute / total
        for ci, ch in enumerate(names):
            cols[f"logabs_{ch}_{band}"] = np.log10(absolute[:, ci] + 1e-12)
            cols[f"rel_{ch}_{band}"] = relative[:, ci]

        for region, members in cfg.REGIONS.items():
            picks = [names.index(c) for c in members if c in names]
            cols[f"region_{region}_{band}"] = (relative[:, picks].mean(axis=1)
                                               if picks else np.zeros(len(data)))
        cols[f"global_{band}"] = relative.mean(axis=1)

    # ---- derived features
    occ = [names.index(c) for c in cfg.OCCIPITAL if c in names]
    eps = 1e-12
    if occ:
        oa = bp["alpha"][:, occ].mean(axis=1)
        ot = bp["theta"][:, occ].mean(axis=1)
        ob = bp["beta"][:, occ].mean(axis=1)
        cols["derived_occ_alpha_theta_logratio"] = np.log10((oa + eps) / (ot + eps))
        cols["derived_occ_alpha_beta_logratio"] = np.log10((oa + eps) / (ob + eps))
    else:
        n = len(data)
        cols["derived_occ_alpha_theta_logratio"] = np.zeros(n)
        cols["derived_occ_alpha_beta_logratio"] = np.zeros(n)

    if "O1" in names and "O2" in names:
        a1 = bp["alpha"][:, names.index("O1")]
        a2 = bp["alpha"][:, names.index("O2")]
        cols["derived_occ_alpha_asymmetry"] = (a2 - a1) / (a2 + a1 + eps)
    else:
        cols["derived_occ_alpha_asymmetry"] = np.zeros(len(data))

    cols["derived_individual_alpha_peak_hz"] = individual_alpha_peak(freqs, psd, occ)

    return pd.DataFrame(cols)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Every numeric feature column, i.e. everything that is not metadata."""
    meta = {"subject", "condition", "run", "epoch_index", "label"}
    return [c for c in df.columns if c not in meta]


def build_feature_matrix(subjects: list[str] | None = None, *,
                         duration: float | None = None,
                         ptp_uv: float | None = None,
                         apply_ica: bool = False,
                         average_reference: bool = True,
                         bands: dict | None = None,
                         verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run preprocessing plus feature extraction over participants.

    Returns ``(features, quality_control)``.  A participant who loses an entire
    condition to artifact rejection is dropped, because the design is paired
    and a one-sided participant cannot contribute to a paired comparison.
    """
    subjects = pp.available_subjects() if subjects is None else subjects
    frames, qc_rows = [], []

    for i, subject in enumerate(subjects, 1):
        per_condition = []
        for run in cfg.RUNS:
            epochs, qc = pp.process_recording(
                subject, run, duration=duration, ptp_uv=ptp_uv,
                apply_ica=apply_ica, average_reference=average_reference)
            qc_rows.append(qc.as_row())
            if epochs is None or len(epochs) == 0:
                per_condition = []
                break
            feats = features_from_epochs(epochs, bands=bands)
            feats.insert(0, "epoch_index", np.arange(len(feats)))
            feats.insert(0, "run", run)
            feats.insert(0, "condition", cfg.RUNS[run])
            feats.insert(0, "subject", subject)
            per_condition.append(feats)

        if len(per_condition) == len(cfg.RUNS):
            frames.extend(per_condition)
        elif verbose:
            print(f"  excluding {subject}: a condition had no surviving epochs")

        if verbose and (i % 10 == 0 or i == len(subjects)):
            print(f"  {i}/{len(subjects)} participants")

    if not frames:
        raise RuntimeError("No usable participants. Check data/raw and rejection settings.")

    features = pd.concat(frames, ignore_index=True)
    features["label"] = (features["condition"] == cfg.POSITIVE_CLASS).astype(int)
    _assert_features_are_live(features)
    return features, pd.DataFrame(qc_rows)


def _assert_features_are_live(features: pd.DataFrame) -> None:
    """Fail loudly if the feature matrix is degenerate.

    A silently-zeroed feature matrix is the worst failure mode this pipeline
    has, because it does not crash: every model simply lands on AUC 0.500 and
    the run looks like a real null result.  That happened once already -- an
    EDF export wrote its physical range in the wrong unit, clipping every
    sample to a hundredth of a microvolt -- and it cost a full pipeline run to
    notice.  A null result must come from the data, never from a unit bug.
    """
    cols = feature_columns(features)
    values = features[cols].to_numpy(dtype=float)

    if not np.isfinite(values).any():
        raise RuntimeError("Feature matrix contains no finite values at all.")

    constant = np.nanstd(values, axis=0) == 0
    if constant.all():
        raise RuntimeError(
            "Every feature is constant across all epochs. The signal was lost "
            "upstream -- check units, filtering, and the EDF physical range."
        )

    power_cols = [i for i, c in enumerate(cols) if c.startswith("logabs_")]
    if power_cols:
        floor = np.isclose(values[:, power_cols], -12.0, atol=1e-6).mean()
        if floor > 0.5:
            raise RuntimeError(
                f"{floor:.0%} of absolute-power features sit on the log floor, "
                "meaning band power is essentially zero. This is a data or unit "
                "problem, not a null result."
            )

    if constant.any():
        n = int(constant.sum())
        print(f"  note: {n}/{len(cols)} features are constant and carry no "
              f"information (kept, but they cannot help any model)")


def main() -> int:
    features, qc = build_feature_matrix()
    features.to_csv(cfg.FEATURES_CSV, index=False)
    qc.to_csv(cfg.QC_CSV, index=False)

    n_feat = len(feature_columns(features))
    print(f"\nfeature matrix -> {cfg.FEATURES_CSV}")
    print(f"  {len(features)} epochs x {n_feat} features")
    print(f"  {features['subject'].nunique()} participants")
    print(features.groupby("condition").size().to_string())
    if features[feature_columns(features)].isna().any().any():
        n = int(features[feature_columns(features)].isna().sum().sum())
        print(f"  WARNING: {n} missing values present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
