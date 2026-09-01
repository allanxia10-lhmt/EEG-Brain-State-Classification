"""EEG preprocessing.

Every step below is a decision that could bias the result, so each one is
documented in place with what it does, why it is needed, the parameter chosen,
and the direction of any bias it could introduce.

The ordering matters.  Bad channels are interpolated *before* the average
reference is applied, otherwise a broken channel is smeared across the whole
montage by the referencing step.

Nothing in this module ever sees the condition label.  Bad-channel detection
and epoch rejection use within-recording statistics only, so they cannot
manufacture a group difference.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class QCRecord:
    """Per-recording quality-control summary."""
    subject: str
    run: str
    condition: str
    sfreq: float = 0.0
    duration_s: float = 0.0
    n_channels: int = 0
    bad_channels: list[str] = field(default_factory=list)
    n_epochs_total: int = 0
    n_epochs_kept: int = 0
    reject_fraction: float = 0.0

    def as_row(self) -> dict:
        return {
            "subject": self.subject, "run": self.run, "condition": self.condition,
            "sfreq": self.sfreq, "duration_s": round(self.duration_s, 2),
            "n_channels": self.n_channels,
            "n_bad_channels": len(self.bad_channels),
            "bad_channels": "|".join(self.bad_channels),
            "n_epochs_total": self.n_epochs_total,
            "n_epochs_kept": self.n_epochs_kept,
            "reject_fraction": round(self.reject_fraction, 4),
        }


# --------------------------------------------------------------- channel names
_CANON: dict[str, str] | None = None


def _canonical_lookup() -> dict[str, str]:
    """Case-insensitive map from a bare electrode label to its montage spelling.

    Built from the montage itself rather than from hand-written rules.  An
    earlier version of this function used a prefix heuristic and produced
    ``FP1``/``FPz``/``FP2``, which standard_1005 spells ``Fp1``/``Fpz``/``Fp2``.
    Those three channels were then silently dropped, which would have removed
    exactly the electrodes the frontal-headband comparison depends on.
    """
    global _CANON
    if _CANON is None:
        import mne

        names = mne.channels.make_standard_montage(cfg.MONTAGE).ch_names
        _CANON = {n.lower(): n for n in names}
    return _CANON


def clean_channel_names(raw) -> None:
    """Map dot-padded EDF labels (``Fc5.``, ``C5..``) onto 10-10 names.

    What it does: strips the padding dots and whitespace, then resolves the
    result against the montage's own spelling, case-insensitively.
    Why: MNE's standard montages key on canonical 10-10 names.  Without this
    no montage attaches, so interpolation and topographic plotting fail, and
    any channel whose case does not match is dropped without warning.
    Bias: none.  This is a relabelling; no sample is touched.
    """
    canon = _canonical_lookup()
    mapping, seen = {}, set()
    for name in raw.ch_names:
        bare = name.strip().replace(" ", "").rstrip(".")
        new = canon.get(bare.lower(), bare)
        if new in seen:                     # never create a duplicate label
            new = name
        seen.add(new)
        mapping[name] = new
    raw.rename_channels(mapping)


def attach_montage(raw) -> None:
    """Attach the standard_1005 montage, dropping channels it cannot place."""
    import mne

    montage = mne.channels.make_standard_montage(cfg.MONTAGE)
    placed = set(montage.ch_names)
    unknown = [ch for ch in raw.ch_names if ch not in placed]
    if unknown:
        raw.drop_channels(unknown)
    raw.set_montage(montage, on_missing="ignore", verbose="error")


# --------------------------------------------------------------- bad channels
def detect_bad_channels(raw) -> list[str]:
    """Flag flat, extreme-variance, and uncorrelated channels.

    Three criteria, all computed within this recording only:

    1. Flat -- standard deviation below ``BAD_FLAT_UV``.  A dead electrode.
    2. Log-variance robust-z above ``BAD_LOGVAR_Z`` (6).  Robust-z uses the
       median and the median absolute deviation, so a couple of bad channels
       cannot inflate the threshold the way a mean and SD would.
    3. Maximum absolute correlation with every other channel below
       ``BAD_MIN_CORR`` (0.30).

    Criterion 3 matters more than it looks.  A first version of this function
    used the variance rule alone and flagged every frontal electrode on the
    first participant, because frontal channels legitimately carry large
    ocular variance -- they were not broken, they were doing their job.  A
    genuinely broken electrode is uncorrelated with its neighbours; a noisy but
    working one is not.

    Bias: interpolating a channel replaces real data with a spatial estimate.
    Over-flagging would smooth away genuine local activity, which is why the
    threshold is deliberately conservative.
    """
    data = raw.get_data()
    names = np.array(raw.ch_names)
    bads: set[str] = set()

    sd = data.std(axis=1)
    bads.update(names[sd < cfg.BAD_FLAT_UV].tolist())

    live = sd >= cfg.BAD_FLAT_UV
    if live.sum() >= 3:
        logvar = np.log(data[live].var(axis=1) + 1e-30)
        med = np.median(logvar)
        mad = np.median(np.abs(logvar - med)) + 1e-12
        rz = 0.6745 * (logvar - med) / mad
        bads.update(names[live][np.abs(rz) > cfg.BAD_LOGVAR_Z].tolist())

        corr = np.corrcoef(data[live])
        np.fill_diagonal(corr, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            max_corr = np.nanmax(np.abs(corr), axis=1)
        bads.update(names[live][max_corr < cfg.BAD_MIN_CORR].tolist())

    return sorted(bads)


# ----------------------------------------------------------------- main filter
def preprocess_raw(raw, apply_ica: bool = False, l_freq: float | None = None,
                   h_freq: float | None = None, average_reference: bool = True):
    """Filter, repair, and reference one recording.

    Band-pass 1-45 Hz, zero-phase FIR.
      * The 1 Hz high-pass removes electrode drift and sweat artifact.  It also
        attenuates the lower delta range, so delta estimates from this pipeline
        are conservative.  That is a real cost, accepted deliberately.
      * The 45 Hz low-pass sits below the 60 Hz US mains frequency, so no notch
        filter is needed at all.
      * Both edges were fixed before any classification was run, and are applied
        identically to both conditions, so the filter cannot create a group
        difference.

    Average reference.
      These recordings have no usable physical reference.  An average reference
      subtracts the mean across electrodes, which attenuates spatially broad
      activity -- in the robustness analysis this turns out to be the single
      preprocessing choice that materially changes the outcome.

    ICA is off by default.  These are 61-second recordings with no EOG or ECG
    channel, so ocular components would have to be identified heuristically
    rather than by correlation with a reference signal, and a misidentified
    component removes genuine frontal activity.  It is available here so the
    robustness analysis can test whether omitting it mattered.
    """
    import mne

    l_freq = cfg.L_FREQ if l_freq is None else l_freq
    h_freq = cfg.H_FREQ if h_freq is None else h_freq

    clean_channel_names(raw)
    attach_montage(raw)
    raw.load_data(verbose="error")
    raw.filter(l_freq, h_freq, method="fir", fir_design=cfg.FILTER_DESIGN,
               phase=cfg.FILTER_PHASE, verbose="error")

    raw.info["bads"] = detect_bad_channels(raw)
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=True, verbose="error")

    if average_reference:
        raw.set_eeg_reference("average", projection=False, verbose="error")

    if apply_ica:
        raw = _apply_ica(raw)
    return raw


def _apply_ica(raw):
    """Remove ocular components using a frontal channel as an EOG proxy.

    Used only by the robustness analysis.  With no true EOG channel the proxy
    is Fp1, which is imperfect: it will also carry genuine frontal brain
    activity, so this can remove signal as well as artifact.
    """
    import mne

    proxy = next((c for c in ("Fp1", "Fpz", "Fp2") if c in raw.ch_names), None)
    if proxy is None:
        return raw
    ica = mne.preprocessing.ICA(n_components=15, random_state=cfg.RANDOM_SEED,
                                max_iter="auto", verbose="error")
    ica.fit(raw, verbose="error")
    try:
        bad, _ = ica.find_bads_eog(raw, ch_name=proxy, verbose="error")
        ica.exclude = bad
    except Exception:
        ica.exclude = []
    return ica.apply(raw.copy(), verbose="error")


# --------------------------------------------------------------- epoching
def make_epochs(raw, duration: float | None = None):
    """Cut the recording into fixed-length non-overlapping windows.

    Non-overlapping keeps successive epochs closer to independent.  Overlapping
    windows would share samples, inflating the apparent sample size without
    adding information -- which then flatters every downstream confidence
    interval.
    """
    import mne

    duration = cfg.EPOCH_SEC if duration is None else duration
    events = mne.make_fixed_length_events(raw, duration=duration,
                                          overlap=cfg.EPOCH_OVERLAP)
    return mne.Epochs(raw, events, tmin=0.0, tmax=duration - 1.0 / raw.info["sfreq"],
                      baseline=None, preload=True, verbose="error")


def reject_epochs(epochs, ptp_uv: float | None = None,
                  chan_frac: float | None = None):
    """Discard epochs where too many channels show a large peak-to-peak swing.

    The conventional rule -- reject if the maximum peak-to-peak across all
    channels exceeds a threshold -- is wrong for a 64-channel montage.  The
    maximum is dominated by whichever single electrode is noisiest, and at the
    usual 150 uV threshold it discarded 97% of epochs in this dataset.

    The rule used instead targets *global* artifacts: an epoch goes only if
    more than ``chan_frac`` of channels exceed ``ptp_uv``.  Movement and muscle
    bursts affect many channels at once; one noisy electrode does not.

    Both values were fixed before any classification was run, using the pooled
    amplitude distribution and one explicit constraint: that rejection rates be
    comparable across conditions, since differential data loss is itself a
    confound.  ``ptp_uv=None`` disables rejection, for the robustness analysis.
    """
    ptp_uv = cfg.REJECT_PTP_UV if ptp_uv is None else ptp_uv
    chan_frac = cfg.REJECT_CHAN_FRAC if chan_frac is None else chan_frac
    data = epochs.get_data(copy=False)
    if data.size == 0:
        return epochs, np.array([], dtype=bool)
    ptp = data.max(axis=2) - data.min(axis=2)          # (n_epochs, n_channels)
    frac_bad = (ptp > ptp_uv * 1e-6).mean(axis=1)
    keep = frac_bad <= chan_frac
    return epochs[np.where(keep)[0]], keep


# --------------------------------------------------------------- driver
def process_recording(subject: str, run: str, *, duration: float | None = None,
                      ptp_uv: float | None = None, apply_ica: bool = False,
                      average_reference: bool = True,
                      l_freq: float | None = None, h_freq: float | None = None):
    """Load one EDF and return ``(epochs, QCRecord)``.  ``epochs`` may be None."""
    import mne

    path = cfg.RAW / subject / f"{subject}{run}.edf"
    condition = cfg.RUNS[run]
    qc = QCRecord(subject=subject, run=run, condition=condition)
    if not path.exists():
        return None, qc

    raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
    qc.sfreq = float(raw.info["sfreq"])
    qc.duration_s = float(raw.n_times / raw.info["sfreq"])

    raw = preprocess_raw(raw, apply_ica=apply_ica, l_freq=l_freq, h_freq=h_freq,
                         average_reference=average_reference)
    qc.n_channels = len(raw.ch_names)
    qc.bad_channels = list(raw.info.get("bads", []))

    epochs = make_epochs(raw, duration=duration)
    qc.n_epochs_total = len(epochs)
    if ptp_uv is not None or cfg.REJECT_PTP_UV is not None:
        epochs, _ = reject_epochs(epochs, ptp_uv=ptp_uv)
    qc.n_epochs_kept = len(epochs)
    qc.reject_fraction = (1.0 - qc.n_epochs_kept / qc.n_epochs_total
                          if qc.n_epochs_total else 0.0)

    if qc.n_epochs_kept == 0:
        return None, qc
    return epochs, qc


def available_subjects() -> list[str]:
    """Participants with both baseline runs present on disk."""
    found = []
    for subject in cfg.SUBJECTS:
        if all((cfg.RAW / subject / f"{subject}{run}.edf").exists()
               for run in cfg.RUNS):
            found.append(subject)
    return found


if __name__ == "__main__":
    import pandas as pd

    subjects = available_subjects()
    print(f"{len(subjects)} participants have both baseline runs.")
    rows = []
    for i, subject in enumerate(subjects, 1):
        for run in cfg.RUNS:
            _, qc = process_recording(subject, run)
            rows.append(qc.as_row())
        if i % 10 == 0 or i == len(subjects):
            print(f"  {i}/{len(subjects)}")
    df = pd.DataFrame(rows)
    df.to_csv(cfg.QC_CSV, index=False)
    print(f"\nquality control -> {cfg.QC_CSV}")
    print(df.groupby("condition")[["n_epochs_kept", "reject_fraction"]].mean())
