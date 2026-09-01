"""Every figure in the paper, generated from results on disk.

Figures are written to figures/ for real data and figures/_smoke/ for synthetic
data, where each one is additionally stamped with a red watermark.  That guard
exists because a plausible-looking figure is the easiest thing in this project
to mistake for a result.

House style: no chartjunk, no 3D, no dual axes, colour used only where it
carries information, and every axis labelled with units.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})

TEAL, CORAL, GREY = "#1b7f79", "#d1495b", "#6b7280"
COND_COLOURS = {"eyes_open": CORAL, "eyes_closed": TEAL}


def _finish(fig, name: str) -> Path:
    """Save a figure, watermarking it when the data are synthetic."""
    fig_dir, _ = cfg.output_dirs()
    if cfg.is_synthetic():
        fig.text(0.5, 0.5, "SYNTHETIC FIXTURE\nNOT A RESULT",
                 ha="center", va="center", fontsize=34, color="red",
                 alpha=0.22, rotation=28, weight="bold", zorder=1000)
    path = fig_dir / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path.name}")
    return path


def _read(results_dir: Path, name: str) -> pd.DataFrame | None:
    path = results_dir / name
    return pd.read_csv(path) if path.exists() else None


# ------------------------------------------------------------------ figure 1
def fig_raw_and_preprocessed(subject: str = "S001") -> Path | None:
    """Raw against preprocessed traces, both conditions, occipital channels."""
    import mne

    from src import preprocessing as pp

    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    for col, (run, cond) in enumerate(cfg.RUNS.items()):
        path = cfg.RAW / subject / f"{subject}{run}.edf"
        if not path.exists():
            plt.close(fig)
            return None
        raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
        pp.clean_channel_names(raw)
        pp.attach_montage(raw)
        picks = [c for c in cfg.OCCIPITAL if c in raw.ch_names]
        seconds, sfreq = 5.0, raw.info["sfreq"]
        n = int(seconds * sfreq)
        t = np.arange(n) / sfreq

        rawd = raw.copy().pick(picks).get_data()[:, :n] * 1e6
        clean = pp.preprocess_raw(raw.copy(), average_reference=True)
        cleand = clean.copy().pick(picks).get_data()[:, :n] * 1e6

        for row, (data, label) in enumerate(((rawd, "Raw"), (cleand, "Preprocessed"))):
            ax = axes[row, col]
            offset = np.nanmax(np.abs(data)) * 1.1 or 1.0
            for i, ch in enumerate(picks):
                ax.plot(t, data[i] + i * offset, lw=0.6,
                        color=COND_COLOURS[cond])
                ax.text(-0.02, i * offset, ch, ha="right", va="center",
                        transform=ax.get_yaxis_transform(), fontsize=8)
            ax.set_yticks([])
            ax.set_title(f"{label} - {cond.replace('_', ' ')}")
            if row == 1:
                ax.set_xlabel("Time (s)")
    fig.suptitle(f"Raw and preprocessed EEG, participant {subject}")
    return _finish(fig, "01_raw_and_preprocessed_eeg.png")


# ------------------------------------------------------------------ figure 2
def fig_power_spectrum(features: pd.DataFrame) -> Path:
    """Occipital spectra by condition, on log and linear axes.

    Reconstructed from band powers rather than raw PSDs, so the shaded alpha
    band is the quantity the classifier actually sees.
    """
    occ = cfg.OCCIPITAL
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    centres = [np.mean(v) for v in cfg.BANDS.values()]

    for ax, logscale in zip(axes, (True, False)):
        for cond in cfg.CONDITIONS:
            sub = features[features["condition"] == cond]
            means, errs = [], []
            for band in cfg.BAND_NAMES:
                cols = [f"logabs_{c}_{band}" for c in occ
                        if f"logabs_{c}_{band}" in features.columns]
                per_subject = sub.groupby("subject")[cols].mean().mean(axis=1)
                means.append(per_subject.mean())
                errs.append(1.96 * per_subject.std(ddof=1) / np.sqrt(len(per_subject)))
            means, errs = np.array(means), np.array(errs)
            vals = means if logscale else 10 ** means
            lo = vals - errs if logscale else 10 ** (means - errs)
            hi = vals + errs if logscale else 10 ** (means + errs)
            ax.plot(centres, vals, "o-", color=COND_COLOURS[cond],
                    label=cond.replace("_", " "))
            ax.fill_between(centres, lo, hi, color=COND_COLOURS[cond], alpha=0.2)
        ax.axvspan(*cfg.BANDS["alpha"], color=GREY, alpha=0.12, zorder=0)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("log10 power (uV^2)" if logscale else "Power (uV^2)")
        ax.set_title("Log scale" if logscale else "Linear scale")
        ax.legend()
    fig.suptitle("Occipital power spectra, mean with 95% CI across participants")
    return _finish(fig, "03_power_spectrum.png")


# ------------------------------------------------------------------ figure 3
def fig_band_comparison(features: pd.DataFrame, bands_table: pd.DataFrame) -> Path:
    """Relative power by condition, absolute effect sizes, and paired lines."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    width, x = 0.36, np.arange(len(cfg.BAND_NAMES))
    for i, cond in enumerate(cfg.CONDITIONS):
        sub = features[features["condition"] == cond]
        means = [sub.groupby("subject")[f"global_{b}"].mean().mean() for b in cfg.BAND_NAMES]
        errs = [1.96 * sub.groupby("subject")[f"global_{b}"].mean().sem() for b in cfg.BAND_NAMES]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=errs, capsize=3,
               color=COND_COLOURS[cond], label=cond.replace("_", " "))
    ax.set_xticks(x); ax.set_xticklabels(cfg.BAND_NAMES)
    ax.set_ylabel("Relative power"); ax.set_title("(a) Whole-scalp relative power")
    ax.legend()

    ax = axes[1]
    colours = [TEAL if v > 0 else CORAL for v in bands_table["abs_dz"]]
    ax.barh(bands_table["band"], bands_table["abs_dz"], color=colours)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Cohen's dz (occipital log absolute power)")
    ax.set_title("(b) Absolute-power effect sizes")

    ax = axes[2]
    cols = [f"rel_{c}_alpha" for c in cfg.OCCIPITAL if f"rel_{c}_alpha" in features.columns]
    tmp = features[["subject", "condition"]].copy()
    tmp["v"] = features[cols].mean(axis=1)
    wide = tmp.groupby(["subject", "condition"])["v"].mean().unstack()
    for _, row in wide.iterrows():
        ax.plot([0, 1], [row["eyes_open"], row["eyes_closed"]],
                color=GREY, alpha=0.35, lw=0.7, marker="o", ms=2)
    ax.plot([0, 1], [wide["eyes_open"].mean(), wide["eyes_closed"].mean()],
            color=TEAL, lw=2.5, marker="o", label="mean")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["eyes open", "eyes closed"])
    ax.set_ylabel("Occipital relative alpha")
    ax.set_title(f"(c) All {len(wide)} participants, paired")
    ax.legend()
    return _finish(fig, "04_band_power_comparison.png")


# ------------------------------------------------------------------ figure 4
def fig_topography(channel_tests: pd.DataFrame) -> Path | None:
    """Scalp maps of the eyes-closed minus eyes-open difference, by band."""
    import mne

    montage = mne.channels.make_standard_montage(cfg.MONTAGE)
    fig, axes = plt.subplots(1, len(cfg.BAND_NAMES), figsize=(14, 3.2))
    for ax, band in zip(np.atleast_1d(axes), cfg.BAND_NAMES):
        sub = channel_tests[channel_tests["band"] == band]
        chans = [c for c in sub["channel"] if c in montage.ch_names]
        if not chans:
            ax.axis("off")
            continue
        values = sub.set_index("channel").loc[chans, "dz"].to_numpy()
        info = mne.create_info(chans, sfreq=cfg.SFREQ, ch_types="eeg", verbose="error")
        info.set_montage(montage, on_missing="ignore", verbose="error")
        lim = float(np.nanmax(np.abs(values))) or 1.0
        mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap="RdBu_r",
                             vlim=(-lim, lim), contours=4)
        ax.set_title(f"{band}\ndz +/-{lim:.2f}", fontsize=9)
    fig.suptitle("Eyes closed minus eyes open, Cohen's dz per channel")
    return _finish(fig, "05_topographic_maps.png")


# ------------------------------------------------------------------ figure 5
def fig_feature_ladder(ladder: pd.DataFrame) -> Path:
    """What feature richness actually buys, against the chance line."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = [f"{r.feature_set.replace('_', ' ')}\n({int(r.n_features)} feat.)"
              for r in ladder.itertuples()]
    aucs = ladder["roc_auc"].to_numpy()
    yerr = None
    if {"roc_auc_ci_low", "roc_auc_ci_high"} <= set(ladder.columns):
        auc_series = ladder["roc_auc"]
        lo = ladder["roc_auc_ci_low"].fillna(auc_series).to_numpy()
        hi = ladder["roc_auc_ci_high"].fillna(auc_series).to_numpy()
        yerr = np.vstack([np.clip(aucs - lo, 0, None), np.clip(hi - aucs, 0, None)])

    ax.bar(labels, aucs, color=TEAL, yerr=yerr, capsize=4)
    for i, v in enumerate(aucs):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    best = float(np.nanmax(aucs))
    ax.axhline(0.5, color=CORAL, ls="--", lw=1, label="chance")
    ax.axhline(best, color="#7c3aed", ls=":", lw=1, label=f"best = {best:.3f}")

    simple = ladder[ladder["feature_set"] == "single_occipital_alpha"]["roc_auc"]
    if len(simple):
        gain = best - float(simple.iloc[0])
        ax.annotate(f"everything above one feature\nbuys {gain:+.3f} AUC",
                    xy=(len(aucs) - 1, best), xytext=(1.35, best - 0.16),
                    fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("ROC-AUC (held-out participants)")
    ax.set_title("What does feature richness actually buy?")
    ax.legend(loc="lower right")
    plt.setp(ax.get_xticklabels(), fontsize=8)
    return _finish(fig, "07b_feature_ladder.png")


# ------------------------------------------------------------------ figure 6
def fig_model_comparison(models: pd.DataFrame, ladder: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    m = models.sort_values("roc_auc")
    yerr = None
    if {"roc_auc_ci_low", "roc_auc_ci_high"} <= set(m.columns):
        v = m["roc_auc"].to_numpy()
        lo = m["roc_auc_ci_low"].fillna(m["roc_auc"]).to_numpy()
        hi = m["roc_auc_ci_high"].fillna(m["roc_auc"]).to_numpy()  # noqa: E501
        yerr = np.vstack([np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)])
    ax.barh(m["model"].str.replace("_", " "), m["roc_auc"], color=TEAL,
            xerr=yerr, capsize=3)
    ax.axvline(0.5, color=CORAL, ls="--", lw=1)
    ax.set_xlim(0.4, 1.0); ax.set_xlabel("ROC-AUC")
    ax.set_title("(a) Model comparison, cluster-bootstrap CIs")

    ax = axes[1]
    ax.plot(range(len(ladder)), ladder["accuracy"], "o-", color=TEAL)
    ax.set_xticks(range(len(ladder)))
    ax.set_xticklabels([s.replace("_", "\n") for s in ladder["feature_set"]], fontsize=7)
    ax.axhline(0.5, color=CORAL, ls="--", lw=1)
    ax.set_ylabel("Accuracy"); ax.set_title("(b) Accuracy across the feature ladder")
    return _finish(fig, "07_model_performance.png")


# ------------------------------------------------------------------ figure 7
def fig_confusion(models: pd.DataFrame) -> Path:
    row = models.set_index("model").loc["logistic_regression"]
    cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=float)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j])}\n{cm[i, j] / cm.sum():.1%}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["open", "closed"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["open", "closed"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Logistic regression\nsens {row['sensitivity']:.3f}, "
                 f"spec {row['specificity']:.3f}")
    ax.grid(False)
    return _finish(fig, "08_confusion_matrix.png")


# ------------------------------------------------------------------ figure 8
def fig_roc(roc: pd.DataFrame, models: pd.DataFrame) -> Path:
    """The ROC curve, which the original figure set omitted."""
    auc = float(models.set_index("model").loc["logistic_regression", "roc_auc"])
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot(roc["fpr"], roc["tpr"], color=TEAL, lw=2,
            label=f"logistic regression (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=CORAL, ls="--", lw=1, label="chance")
    ax.fill_between(roc["fpr"], roc["tpr"], roc["fpr"], color=TEAL, alpha=0.12)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC, held-out participants")
    ax.legend(loc="lower right")
    return _finish(fig, "09_roc_curve.png")


# ------------------------------------------------------------------ figure 9
def fig_feature_importance(imp: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    top = imp.head(15).iloc[::-1]
    axes[0].barh(top["feature"], top["importance_mean"],
                 xerr=top["importance_std"], color=TEAL, capsize=2)
    axes[0].set_xlabel("Permutation importance (AUC drop)")
    axes[0].set_title("(a) Top 15 features")
    plt.setp(axes[0].get_yticklabels(), fontsize=6)

    by_band = imp.groupby("band")["importance_mean"].sum().sort_values(ascending=False)
    axes[1].bar(by_band.index, by_band.to_numpy(), color=TEAL)
    axes[1].set_ylabel("Summed importance"); axes[1].set_title("(b) By band")

    by_ch = (imp[imp["channel"] != "-"].groupby("channel")["importance_mean"]
             .sum().sort_values(ascending=False).head(15).iloc[::-1])
    axes[2].barh(by_ch.index, by_ch.to_numpy(), color=TEAL)
    axes[2].set_xlabel("Summed importance"); axes[2].set_title("(c) Top 15 electrodes")
    fig.suptitle("Permutation importance measures redundancy, not relevance, "
                 "on correlated features", fontsize=9)
    return _finish(fig, "10_feature_importance.png")


# ----------------------------------------------------------------- figure 10
def fig_subject_level(per_subject: pd.DataFrame, loso: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    s = per_subject.sort_values("accuracy")
    below = s["accuracy"] < 0.60
    axes[0].bar(range(len(s)), s["accuracy"],
                color=np.where(below, CORAL, TEAL))
    axes[0].axhline(0.5, color="black", ls="--", lw=1, label="chance")
    axes[0].axhline(0.6, color=GREY, ls=":", lw=1, label="0.60")
    axes[0].set_xlabel("Participant (sorted)"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("(a) Per-participant accuracy"); axes[0].legend()

    axes[1].hist(per_subject["accuracy"], bins=18, color=TEAL, edgecolor="white")
    axes[1].axvline(per_subject["accuracy"].mean(), color=CORAL, lw=2,
                    label=f"mean {per_subject['accuracy'].mean():.3f}")
    axes[1].axvline(0.5, color="black", ls="--", lw=1)
    n_below = int(below.sum())
    axes[1].set_xlabel("Accuracy"); axes[1].set_ylabel("Participants")
    axes[1].set_title(f"(b) Distribution: {n_below}/{len(s)} below 0.60")
    axes[1].legend()
    return _finish(fig, "11_subject_level_performance.png")


# ----------------------------------------------------------------- figure 11
def fig_tradeoff(ablation: pd.DataFrame, windows: pd.DataFrame,
                 surface: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    # Two electrode sets can share a count (three occipital, three frontal), so
    # a single line through every point zigzags vertically and reads as noise.
    # The posterior progression is the trend; the headband is a separate claim.
    posterior = ablation[~ablation["electrode_set"].str.startswith("headband")]
    headband = ablation[ablation["electrode_set"].str.startswith("headband")]
    p = posterior.sort_values("n_electrodes")
    ax.plot(p["n_electrodes"], p["roc_auc"], "o-", color=TEAL,
            label="informed posterior placement")
    if len(headband):
        ax.plot(headband["n_electrodes"], headband["roc_auc"], "D",
                color=CORAL, ms=7, label="frontal headband")
    if "random_auc_mean" in ablation.columns:
        r = p.dropna(subset=["random_auc_mean"]).sort_values("n_electrodes")
        ax.plot(r["n_electrodes"], r["random_auc_mean"], "s--", color=GREY,
                label="random same-size")
        if {"random_auc_min", "random_auc_max"} <= set(r.columns):
            ax.fill_between(r["n_electrodes"], r["random_auc_min"],
                            r["random_auc_max"], color=GREY, alpha=0.18)
    ax.set_xscale("log"); ax.set_xlabel("Electrodes (log scale)")
    ax.set_ylabel("ROC-AUC"); ax.set_title("(a) Electrode ablation")
    ax.legend(fontsize=7)

    ax = axes[1]
    for label, grp in surface.groupby("electrode_set"):
        grp = grp.sort_values("window_s")
        ax.plot(grp["window_s"], grp["roc_auc"], "o-", lw=1.2, ms=3, label=label)
    ax.set_xscale("log", base=2); ax.set_xlabel("Window length (s)")
    ax.set_ylabel("ROC-AUC"); ax.set_title("(b) Recording time")
    ax.legend(fontsize=7)

    ax = axes[2]
    pivot = surface.pivot(index="electrode_set", columns="window_s", values="roc_auc")
    # Explicit configured order, so that ties on electrode count do not shuffle
    # rows between runs and the headband stays at the bottom where it belongs.
    order = [k for k in cfg.ELECTRODE_SETS if k in pivot.index]
    order += [k for k in pivot.index if k not in order]
    pivot = pivot.reindex(order)
    im = ax.imshow(pivot.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:g}s" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.to_numpy()[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=6, color="white")
    fig.colorbar(im, ax=ax, label="ROC-AUC")
    ax.set_title("(c) Trade-off surface"); ax.grid(False)
    return _finish(fig, "06_complexity_tradeoff.png")


# ----------------------------------------------------------------- figure 12
def fig_robustness(rob: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{r.analysis}: {r.setting}" for r in rob.itertuples()]
    y = np.arange(len(rob))
    ax.barh(y, rob["roc_auc"], color=[TEAL if a != "referencing" else CORAL
                                      for a in rob["analysis"]])
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    lo, hi = rob["roc_auc"].min(), rob["roc_auc"].max()
    ax.set_xlim(max(0.4, lo - 0.05), min(1.0, hi + 0.03))
    for i, v in enumerate(rob["roc_auc"]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=6)
    ax.set_xlabel("ROC-AUC")
    ax.set_title(f"Robustness across {len(rob)} configurations "
                 f"(span {hi - lo:.3f})")
    return _finish(fig, "12_robustness_analysis.png")


# ---------------------------------------------------------------------- main
def main() -> int:
    _, results_dir = cfg.output_dirs()
    fig_dir, _ = cfg.output_dirs()
    print(f"Writing figures to {fig_dir}")
    if cfg.is_synthetic():
        print("  SYNTHETIC MODE: every figure will be watermarked")

    features = pd.read_csv(cfg.FEATURES_CSV)
    made = []

    made.append(fig_raw_and_preprocessed())
    made.append(fig_power_spectrum(features))

    bands = _read(results_dir, "band_comparison.csv")
    if bands is not None:
        made.append(fig_band_comparison(features, bands))
    chan = _read(results_dir, "channel_band_tests_logabs.csv")
    if chan is not None:
        made.append(fig_topography(chan))

    ladder = _read(results_dir, "feature_ladder.csv")
    models = _read(results_dir, "model_comparison.csv")
    if ladder is not None:
        made.append(fig_feature_ladder(ladder))
    if ladder is not None and models is not None:
        made.append(fig_model_comparison(models, ladder))
    if models is not None:
        made.append(fig_confusion(models))
    roc = _read(results_dir, "roc_curve.csv")
    if roc is not None and models is not None:
        made.append(fig_roc(roc, models))

    imp = _read(results_dir, "feature_importance.csv")
    if imp is not None:
        made.append(fig_feature_importance(imp))

    per_subject = _read(results_dir, "per_subject_accuracy.csv")
    loso = _read(results_dir, "loso_summary.csv")
    if per_subject is not None:
        made.append(fig_subject_level(per_subject, loso))

    ablation = _read(results_dir, "electrode_ablation.csv")
    windows = _read(results_dir, "window_sweep.csv")
    surface = _read(results_dir, "tradeoff_surface.csv")
    if ablation is not None and windows is not None and surface is not None:
        made.append(fig_tradeoff(ablation, windows, surface))

    rob = _read(results_dir, "robustness.csv")
    if rob is not None:
        made.append(fig_robustness(rob))

    print(f"\n{len([m for m in made if m])} figures written to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
