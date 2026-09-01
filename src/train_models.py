"""Model training and evaluation, with participant-level validation throughout.

The leakage problem
-------------------
Each participant contributes roughly fifty epochs that share their skull
thickness, electrode placement, alpha peak frequency, and overall amplitude.  A
model given epochs from the same person in both training and test data can
identify the *person* and recall their condition, rather than learning anything
about brain state.  It will look excellent and generalise to nobody.

Every split in this module is therefore made over participants:

  * the held-out set comes from GroupShuffleSplit over participants and is not
    touched until final evaluation;
  * hyperparameter search uses GroupKFold over training participants only;
  * all scaling is fitted inside folds, via a Pipeline, never on the full data;
  * ``assert_no_leakage`` is called on every split and raises rather than warns.

``leaky_baseline`` deliberately does it wrong, splitting epochs at random, so
the paper can quantify how much that single error inflates the result.  It is
the only function here that is allowed to be wrong, and it says so.

Confidence intervals use a cluster bootstrap that resamples participants, not
epochs.  An ordinary bootstrap over epochs would treat ~50 dependent
observations as independent and report intervals several times too narrow.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402
from src.feature_extraction import feature_columns  # noqa: E402


# ------------------------------------------------------------------ guards
def assert_no_leakage(groups_train: np.ndarray, groups_test: np.ndarray) -> None:
    """Raise if any participant appears on both sides of a split."""
    overlap = set(np.unique(groups_train)) & set(np.unique(groups_test))
    if overlap:
        raise AssertionError(
            f"Participant-level leakage: {sorted(overlap)[:5]} appear in both "
            f"training and test partitions ({len(overlap)} total)."
        )


# ------------------------------------------------------------------ models
def make_models(seed: int = cfg.RANDOM_SEED) -> dict[str, Pipeline]:
    """Progressively more capable classifiers, all behind the same scaler.

    Scaling lives inside the Pipeline so that GridSearchCV refits it on each
    training fold.  Fitting a scaler on the full dataset before splitting is
    one of the commonest and quietest forms of leakage.
    """
    return {
        "dummy": Pipeline([
            ("scale", StandardScaler()),
            ("clf", DummyClassifier(strategy="most_frequent"))]),
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=seed))]),
        "linear_svm": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LinearSVC(random_state=seed, max_iter=20000, dual="auto"))]),
        "random_forest": Pipeline([
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=seed, n_jobs=-1))]),
        "hist_gradient_boosting": Pipeline([
            ("scale", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(random_state=seed))]),
    }


def decision_scores(model, X: np.ndarray) -> np.ndarray:
    """Continuous scores for ROC-AUC, whichever interface the model offers."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)


# ------------------------------------------------------------------ splitting
def load_features(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or cfg.FEATURES_CSV)
    if "label" not in df.columns:
        df["label"] = (df["condition"] == cfg.POSITIVE_CLASS).astype(int)
    return df


def split_by_participant(df: pd.DataFrame, test_size: float = cfg.TEST_SIZE,
                         seed: int = cfg.RANDOM_SEED):
    """Hold out a fraction of PARTICIPANTS, not a fraction of epochs."""
    groups = df["subject"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, df["label"], groups))
    assert_no_leakage(groups[train_idx], groups[test_idx])
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def xy(df: pd.DataFrame, columns: list[str] | None = None):
    cols = columns or feature_columns(df)
    X = df[cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, df["label"].to_numpy(), df["subject"].to_numpy(), cols


# ------------------------------------------------------------------ metrics
def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             y_score: np.ndarray | None = None) -> dict:
    """Accuracy alone is not a result.  Report the whole confusion structure."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y_true)),
    }
    out["roc_auc"] = (roc_auc_score(y_true, y_score)
                      if y_score is not None and len(np.unique(y_true)) > 1 else 0.5)
    return out


def cluster_bootstrap_ci(y_true, y_score, y_pred, groups,
                         n_boot: int = cfg.N_BOOTSTRAP,
                         seed: int = cfg.RANDOM_SEED) -> dict:
    """Percentile CIs from resampling PARTICIPANTS with replacement.

    Resampling epochs would treat dependent observations as independent, and
    the resulting intervals would be far too narrow to be honest.
    """
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    y_pred, groups = np.asarray(y_pred), np.asarray(groups)
    unique = np.unique(groups)
    index = {g: np.where(groups == g)[0] for g in unique}

    accs, aucs = [], []
    for _ in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index[g] for g in picked])
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        if len(np.unique(y_true[idx])) > 1:
            aucs.append(roc_auc_score(y_true[idx], y_score[idx]))

    def pct(v):
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)

    lo_a, hi_a = pct(accs)
    lo_u, hi_u = pct(aucs)
    return {"accuracy_ci_low": lo_a, "accuracy_ci_high": hi_a,
            "roc_auc_ci_low": lo_u, "roc_auc_ci_high": hi_u,
            "n_bootstrap": n_boot, "n_groups": len(unique)}


# ------------------------------------------------------------------ training
def fit_tuned(name: str, train: pd.DataFrame, columns: list[str] | None = None,
              n_folds: int = cfg.N_CV_FOLDS, seed: int = cfg.RANDOM_SEED):
    """Grid-search a model using GroupKFold over TRAINING participants only.

    The held-out set is never seen here.  Optimising against it, even once,
    would turn the final number into a training metric wearing a disguise.
    """
    X, y, groups, cols = xy(train, columns)
    model = make_models(seed)[name]
    grid = cfg.PARAM_GRIDS.get(name)
    if not grid:
        model.fit(X, y)
        return model, {}, cols

    n_splits = min(n_folds, len(np.unique(groups)))
    search = GridSearchCV(model, grid, cv=GroupKFold(n_splits=n_splits),
                          scoring="roc_auc", n_jobs=-1, refit=True)
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_, cols


def train_and_evaluate(train: pd.DataFrame, test: pd.DataFrame,
                       columns: list[str] | None = None,
                       models: list[str] | None = None,
                       n_boot: int = cfg.N_BOOTSTRAP,
                       verbose: bool = True) -> pd.DataFrame:
    """Fit every model on training participants, score on held-out ones."""
    assert_no_leakage(train["subject"].to_numpy(), test["subject"].to_numpy())
    names = models or list(make_models())
    rows = []
    for name in names:
        model, params, cols = fit_tuned(name, train, columns)
        Xte, yte, gte, _ = xy(test, cols)
        y_pred = model.predict(Xte)
        y_score = decision_scores(model, Xte)
        row = {"model": name, **evaluate(yte, y_pred, y_score),
               "n_features": len(cols), "best_params": str(params)}
        if n_boot:
            row.update(cluster_bootstrap_ci(yte, y_score, y_pred, gte, n_boot))
        rows.append(row)
        if verbose:
            print(f"  {name:24s} acc {row['accuracy']:.3f}  auc {row['roc_auc']:.3f}")
    return pd.DataFrame(rows)


def leave_one_subject_out(df: pd.DataFrame, columns: list[str] | None = None,
                          model_name: str = "logistic_regression",
                          C: float = cfg.LOGREG_C_FIXED,
                          verbose: bool = True) -> tuple[dict, pd.DataFrame]:
    """Leave-one-subject-out cross-validation over every participant.

    A more demanding generalisation estimate than a single held-out split, and
    the one that exposes the per-participant tail: an aggregate figure can look
    healthy while a substantial minority of participants classify near chance.
    """
    subjects = df["subject"].unique()
    cols = columns or feature_columns(df)
    y_true_all, y_score_all, y_pred_all, group_all = [], [], [], []
    per_subject = []

    for i, subject in enumerate(subjects, 1):
        train = df[df["subject"] != subject]
        test = df[df["subject"] == subject]
        assert_no_leakage(train["subject"].to_numpy(), test["subject"].to_numpy())
        Xtr, ytr, _, _ = xy(train, cols)
        Xte, yte, _, _ = xy(test, cols)
        if len(np.unique(ytr)) < 2:
            continue
        model = Pipeline([("scale", StandardScaler()),
                          ("clf", LogisticRegression(C=C, max_iter=5000,
                                                     random_state=cfg.RANDOM_SEED))])
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        score = decision_scores(model, Xte)
        per_subject.append({"subject": subject, "n_epochs": len(yte),
                            "accuracy": accuracy_score(yte, pred)})
        y_true_all.append(yte); y_pred_all.append(pred)
        y_score_all.append(score); group_all.append(np.repeat(subject, len(yte)))
        if verbose and (i % 20 == 0 or i == len(subjects)):
            print(f"    LOSO {i}/{len(subjects)}")

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_score = np.concatenate(y_score_all)
    groups = np.concatenate(group_all)
    per_subject_df = pd.DataFrame(per_subject)

    summary = {"model": f"{model_name}_LOSO", **evaluate(y_true, y_pred, y_score),
               "mean_subject_accuracy": float(per_subject_df["accuracy"].mean()),
               "std_subject_accuracy": float(per_subject_df["accuracy"].std(ddof=1)),
               "min_subject_accuracy": float(per_subject_df["accuracy"].min()),
               "median_subject_accuracy": float(per_subject_df["accuracy"].median()),
               "n_subjects_below_0.60": int((per_subject_df["accuracy"] < 0.60).sum()),
               "n_subjects": len(per_subject_df)}
    return summary, per_subject_df


def leaky_baseline(df: pd.DataFrame, columns: list[str] | None = None,
                   seed: int = cfg.RANDOM_SEED) -> dict:
    """Deliberately WRONG: split epochs at random, ignoring participant identity.

    This exists solely to quantify the inflation that participant-level
    splitting prevents.  Its output is a measurement of a methodological error,
    never a performance claim.
    """
    from sklearn.model_selection import train_test_split

    cols = columns or feature_columns(df)
    X, y, _, _ = xy(df, cols)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg.TEST_SIZE,
                                          random_state=seed, stratify=y)
    model = Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(C=cfg.LOGREG_C_FIXED,
                                                 max_iter=5000, random_state=seed))])
    model.fit(Xtr, ytr)
    out = evaluate(yte, model.predict(Xte), decision_scores(model, Xte))
    out["model"] = "logistic_regression_LEAKY_epoch_split"
    out["warning"] = "Epoch-level split. Inflated by design. Not a result."
    return out


def roc_points(df_train: pd.DataFrame, df_test: pd.DataFrame,
               columns: list[str] | None = None) -> pd.DataFrame:
    """Points for the ROC curve figure."""
    model, _, cols = fit_tuned("logistic_regression", df_train, columns)
    Xte, yte, _, _ = xy(df_test, cols)
    fpr, tpr, thr = roc_curve(yte, decision_scores(model, Xte))
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})


# ---------------------------------------------------------------------- main
def main() -> int:
    _, results_dir = cfg.output_dirs()
    df = load_features()
    train, test = split_by_participant(df)
    print(f"{train['subject'].nunique()} training participants, "
          f"{test['subject'].nunique()} held out "
          f"({len(train)} / {len(test)} epochs)")

    print("\nModel comparison, all features:")
    table = train_and_evaluate(train, test)
    table.to_csv(results_dir / "model_comparison.csv", index=False)

    print("\nLeave-one-subject-out:")
    loso, per_subject = leave_one_subject_out(df)
    pd.DataFrame([loso]).to_csv(results_dir / "loso_summary.csv", index=False)
    per_subject.to_csv(results_dir / "per_subject_accuracy.csv", index=False)
    print(f"  accuracy {loso['mean_subject_accuracy']:.3f} "
          f"+/- {loso['std_subject_accuracy']:.3f}, auc {loso['roc_auc']:.3f}, "
          f"{loso['n_subjects_below_0.60']}/{loso['n_subjects']} below 0.60")

    leaky = leaky_baseline(df)
    honest = table.set_index("model").loc["logistic_regression"]
    pd.DataFrame([leaky]).to_csv(results_dir / "leakage_demonstration.csv", index=False)
    print(f"\nLeakage check: participant split auc {honest['roc_auc']:.3f} "
          f"vs epoch split auc {leaky['roc_auc']:.3f} "
          f"(inflation {leaky['roc_auc'] - honest['roc_auc']:+.3f})")

    roc_points(train, test).to_csv(results_dir / "roc_curve.csv", index=False)
    print(f"\nresults -> {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
