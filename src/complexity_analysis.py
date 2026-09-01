"""The complexity sweep: what does each additional resource actually buy?

Four questions, each answered by holding everything else fixed:

  * the feature ladder -- dummy, one occipital alpha feature, five whole-scalp
    band powers, 320 per-channel relative powers, all 674;
  * electrode ablation -- 64 down to one, each compared against random
    same-size subsets so that informed placement can be judged against
    arbitrary placement;
  * window length -- 0.5 s to 8 s;
  * the crossed surface, which shows whether recording time substitutes for
    electrodes.

The learner is held fixed at logistic regression with the pre-selected C
throughout the sweep.  Re-tuning per configuration would confound "this
configuration carries more information" with "this configuration got a luckier
hyperparameter search", and the whole point is to attribute the difference to
the data rather than to the fitting procedure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402
from src import train_models as tm  # noqa: E402
from src.feature_extraction import feature_columns  # noqa: E402


def _fixed_model(seed: int = cfg.RANDOM_SEED) -> Pipeline:
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=cfg.LOGREG_C_FIXED,
                                                max_iter=5000, random_state=seed))])


def score_columns(train: pd.DataFrame, test: pd.DataFrame, columns: list[str],
                  n_boot: int = 0) -> dict:
    """Fit the fixed model on a chosen feature subset and score the held-out set."""
    tm.assert_no_leakage(train["subject"].to_numpy(), test["subject"].to_numpy())
    Xtr, ytr, _, _ = tm.xy(train, columns)
    Xte, yte, gte, _ = tm.xy(test, columns)
    model = _fixed_model()
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    score = tm.decision_scores(model, Xte)
    out = {"n_features": len(columns), **tm.evaluate(yte, pred, score)}
    if n_boot:
        out.update(tm.cluster_bootstrap_ci(yte, score, pred, gte, n_boot))
    return out


# ------------------------------------------------------------- feature ladder
def feature_ladder(train: pd.DataFrame, test: pd.DataFrame,
                   n_boot: int = cfg.N_BOOTSTRAP) -> pd.DataFrame:
    """Progressively richer feature sets, same learner throughout.

    The rung that matters is the first one: a single occipital relative alpha
    value, which is the feature Berger could have computed by eye in 1929.
    Everything above it has to justify its cost against that baseline.
    """
    all_cols = feature_columns(train)
    ladders = {
        "single_occipital_alpha": [c for c in ["rel_Oz_alpha"] if c in all_cols],
        "whole_scalp_band_powers": [f"global_{b}" for b in cfg.BAND_NAMES
                                    if f"global_{b}" in all_cols],
        "per_channel_relative": [c for c in all_cols if c.startswith("rel_")],
        "all_features": all_cols,
    }
    rows = []

    Xte, yte, gte, _ = tm.xy(test, all_cols)
    dummy = _fixed_model()
    from sklearn.dummy import DummyClassifier
    dummy = Pipeline([("scale", StandardScaler()),
                      ("clf", DummyClassifier(strategy="most_frequent"))])
    Xtr, ytr, _, _ = tm.xy(train, all_cols)
    dummy.fit(Xtr, ytr)
    rows.append({"feature_set": "dummy_baseline", "n_features": 0,
                 **tm.evaluate(yte, dummy.predict(Xte),
                               np.full(len(yte), 0.5))})

    for name, cols in ladders.items():
        if not cols:
            continue
        rows.append({"feature_set": name, **score_columns(train, test, cols, n_boot)})
    return pd.DataFrame(rows)


# --------------------------------------------------------- electrode ablation
def _cols_for_channels(all_cols: list[str], channels: list[str] | None) -> list[str]:
    """Feature columns belonging to a given electrode subset.

    Region and global summaries are excluded from subsets, because they are
    computed across the whole montage and would smuggle information from
    electrodes the subset does not have.
    """
    if channels is None:
        return all_cols
    keep = []
    for col in all_cols:
        parts = col.split("_")
        if parts[0] in {"logabs", "rel"} and len(parts) >= 3:
            if "_".join(parts[1:-1]) in channels:
                keep.append(col)
    return keep


def all_channels(df: pd.DataFrame) -> list[str]:
    names = set()
    for col in df.columns:
        parts = col.split("_")
        if parts[0] == "rel" and len(parts) >= 3:
            names.add("_".join(parts[1:-1]))
    return sorted(names)


def electrode_ablation(train: pd.DataFrame, test: pd.DataFrame,
                       n_random: int = cfg.N_RANDOM_SUBSETS,
                       seed: int = cfg.RANDOM_SEED) -> pd.DataFrame:
    """Named electrode budgets, each against random same-size subsets.

    The random comparison is the informative half.  Knowing that three
    occipital electrodes reach AUC 0.83 means little on its own; knowing that
    three *random* electrodes reach 0.80 tells you what placement is worth.
    """
    all_cols = feature_columns(train)
    channels = all_channels(train)
    rng = np.random.default_rng(seed)
    rows = []

    for label, chans in cfg.ELECTRODE_SETS.items():
        present = channels if chans is None else [c for c in chans if c in channels]
        if not present:
            continue
        cols = _cols_for_channels(all_cols, None if chans is None else present)
        res = score_columns(train, test, cols)
        row = {"electrode_set": label, "n_electrodes": len(present),
               "channels": "|".join(present), **res}

        if chans is not None and len(present) < len(channels):
            aucs = []
            for _ in range(n_random):
                pick = list(rng.choice(channels, size=len(present), replace=False))
                aucs.append(score_columns(train, test,
                                          _cols_for_channels(all_cols, pick))["roc_auc"])
            row.update({"random_auc_mean": float(np.mean(aucs)),
                        "random_auc_min": float(np.min(aucs)),
                        "random_auc_max": float(np.max(aucs)),
                        "placement_advantage": float(res["roc_auc"] - np.mean(aucs))})
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------- window length
def window_sweep(subjects: list[str] | None = None,
                 windows: list[float] | None = None,
                 electrode_sets: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-extract features at each window length and score the same split.

    This is the expensive analysis: every window length requires a full
    re-epoching and re-extraction, because a 4-second epoch is not obtainable
    by pooling 2-second ones.  A short recording also yields few long windows,
    so participants can drop out at the long end -- that attrition is reported
    rather than hidden.
    """
    from src.feature_extraction import build_feature_matrix

    windows = windows or cfg.WINDOW_SWEEP
    electrode_sets = electrode_sets or cfg.ELECTRODE_SETS
    flat_rows, surface_rows = [], []

    for win in windows:
        print(f"  window {win:.1f}s: extracting features")
        feats, _ = build_feature_matrix(subjects, duration=win, verbose=False)
        train, test = tm.split_by_participant(feats)
        all_cols = feature_columns(feats)
        channels = all_channels(feats)

        full = score_columns(train, test, all_cols, n_boot=500)
        flat_rows.append({"window_s": win, "n_subjects": feats["subject"].nunique(),
                          "n_test_epochs": len(test), **full})

        for label, chans in electrode_sets.items():
            present = channels if chans is None else [c for c in chans if c in channels]
            if not present:
                continue
            cols = _cols_for_channels(all_cols, None if chans is None else present)
            res = score_columns(train, test, cols)
            surface_rows.append({"window_s": win, "electrode_set": label,
                                 "n_electrodes": len(present),
                                 "roc_auc": res["roc_auc"],
                                 "accuracy": res["accuracy"]})
    return pd.DataFrame(flat_rows), pd.DataFrame(surface_rows)


# ---------------------------------------------------------- feature importance
def permutation_importance_table(train: pd.DataFrame, test: pd.DataFrame,
                                 n_repeats: int = 5,
                                 seed: int = cfg.RANDOM_SEED) -> pd.DataFrame:
    """Permutation importance, with the caveat that makes it interpretable.

    On correlated features this measures REDUNDANCY, not relevance.  Alpha
    information is duplicated across dozens of posterior electrodes, so
    permuting any single alpha feature barely hurts the model -- it simply
    reads the same information off a neighbour.  A band can therefore look
    unimportant here while carrying the largest univariate effect in the
    dataset.  The feature-family ablation in robustness.py answers the question
    this table cannot.
    """
    from sklearn.inspection import permutation_importance

    cols = feature_columns(train)
    Xtr, ytr, _, _ = tm.xy(train, cols)
    Xte, yte, _, _ = tm.xy(test, cols)
    model = _fixed_model()
    model.fit(Xtr, ytr)
    result = permutation_importance(model, Xte, yte, n_repeats=n_repeats,
                                    random_state=seed, scoring="roc_auc", n_jobs=-1)

    rows = []
    for col, mean, std in zip(cols, result.importances_mean, result.importances_std):
        parts = col.split("_")
        band = parts[-1] if parts[-1] in cfg.BAND_NAMES else "derived"
        channel = "_".join(parts[1:-1]) if parts[0] in {"logabs", "rel"} else "-"
        rows.append({"feature": col, "channel": channel, "band": band,
                     "importance_mean": float(mean), "importance_std": float(std)})
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------- main
def main() -> int:
    _, results_dir = cfg.output_dirs()
    df = tm.load_features()
    train, test = tm.split_by_participant(df)

    print("Feature ladder:")
    ladder = feature_ladder(train, test)
    ladder.to_csv(results_dir / "feature_ladder.csv", index=False)
    print(ladder[["feature_set", "n_features", "accuracy", "roc_auc"]].to_string(index=False))

    print("\nElectrode ablation:")
    ablation = electrode_ablation(train, test)
    ablation.to_csv(results_dir / "electrode_ablation.csv", index=False)
    show = [c for c in ["electrode_set", "n_electrodes", "roc_auc",
                        "random_auc_mean", "placement_advantage"] if c in ablation.columns]
    print(ablation[show].to_string(index=False))

    print("\nWindow-length sweep (re-extracts features at each length):")
    windows, surface = window_sweep()
    windows.to_csv(results_dir / "window_sweep.csv", index=False)
    surface.to_csv(results_dir / "tradeoff_surface.csv", index=False)
    print(windows[["window_s", "n_subjects", "n_test_epochs", "accuracy", "roc_auc"]]
          .to_string(index=False))

    print("\nPermutation importance:")
    imp = permutation_importance_table(train, test)
    imp.to_csv(results_dir / "feature_importance.csv", index=False)
    by_band = imp.groupby("band")["importance_mean"].sum().sort_values(ascending=False)
    by_band.to_csv(results_dir / "feature_importance_by_band.csv")
    print(by_band.to_string())
    print("  (permutation importance on correlated features measures redundancy, "
          "not relevance -- see robustness.py for the ablation)")

    print(f"\nresults -> {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
