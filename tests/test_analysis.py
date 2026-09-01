"""Feature, statistics, leakage, and reproducibility tests."""
from __future__ import annotations

import numpy as np
import pytest

from src import config as cfg
from src import feature_extraction as fe
from src import statistics as st
from src import train_models as tm


# ------------------------------------------------------------------ features
def test_feature_matrix_has_the_documented_674_columns(epochs_pair):
    df = fe.features_from_epochs(epochs_pair["eyes_closed"])
    assert df.shape[1] == 674
    counts = {p: sum(c.startswith(p + "_") for c in df.columns)
              for p in ("logabs", "rel", "region", "global", "derived")}
    assert counts == {"logabs": 320, "rel": 320, "region": 25,
                      "global": 5, "derived": 4}


def test_features_contain_no_missing_or_infinite_values(epochs_pair):
    df = fe.features_from_epochs(epochs_pair["eyes_open"])
    assert not df.isna().to_numpy().any()
    assert np.isfinite(df.to_numpy(dtype=float)).all()


def test_relative_band_powers_sum_to_one_per_channel(epochs_pair):
    """Relative power is compositional, which is why absolute power is kept."""
    df = fe.features_from_epochs(epochs_pair["eyes_open"])
    for ch in ("Oz", "Cz", "Fpz"):
        cols = [f"rel_{ch}_{b}" for b in cfg.BAND_NAMES]
        assert np.allclose(df[cols].sum(axis=1).to_numpy(), 1.0, atol=1e-6)


def test_band_power_integrates_a_known_sine_wave():
    """A 10 Hz sine must put its power in alpha and nowhere else."""
    sfreq, n = 160.0, 320
    t = np.arange(n) / sfreq
    data = (50e-6 * np.sin(2 * np.pi * 10 * t))[None, None, :]
    freqs, psd = fe.epoch_psd(data, sfreq)
    bp = fe.band_power(freqs, psd)
    total = sum(float(v.ravel()[0]) for v in bp.values())
    assert float(bp["alpha"].ravel()[0]) / total > 0.9


def test_individual_alpha_peak_recovers_a_planted_frequency():
    sfreq, n = 160.0, 640
    t = np.arange(n) / sfreq
    data = (40e-6 * np.sin(2 * np.pi * 11 * t))[None, None, :]
    freqs, psd = fe.epoch_psd(data, sfreq)
    peak = fe.individual_alpha_peak(freqs, psd, [0])
    assert abs(float(peak[0]) - 11.0) <= 1.0


def test_degenerate_feature_matrix_raises_rather_than_scoring_chance(small_feature_table):
    """A zeroed feature matrix must fail loudly, not look like a null result."""
    df = small_feature_table.copy()
    for col in fe.feature_columns(df):
        df[col] = -12.0 if col.startswith("logabs_") else 0.0
    with pytest.raises(RuntimeError):
        fe._assert_features_are_live(df)


def test_feature_columns_excludes_metadata(small_feature_table):
    cols = fe.feature_columns(small_feature_table)
    assert not ({"subject", "condition", "label", "epoch_index"} & set(cols))
    assert len(cols) == 674


# ---------------------------------------------------------------- statistics
def test_cohens_dz_matches_a_hand_computed_value():
    diff = np.array([1.0, 2.0, 3.0, 4.0])
    assert st.cohens_dz(diff) == pytest.approx(diff.mean() / diff.std(ddof=1))


def test_benjamini_hochberg_is_less_conservative_than_bonferroni():
    p = np.array([0.001, 0.008, 0.02, 0.04, 0.3])
    rejected, adj = st.benjamini_hochberg(p, q=0.05)
    assert rejected.sum() >= (p < 0.05 / len(p)).sum()
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12), "adjusted p must be monotone"


def test_paired_test_recovers_a_known_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 60)
    b = a + 0.8
    res = st.paired_test(a, b)
    assert res["mean_diff"] == pytest.approx(0.8, abs=1e-9)
    assert res["p_ttest"] < 1e-10
    assert res["n_positive"] == 60


def test_paired_ci_brackets_the_mean_difference():
    rng = np.random.default_rng(1)
    diff = rng.normal(0.5, 1.0, 100)
    lo, hi = st.paired_ci(diff)
    assert lo < diff.mean() < hi


def test_occipital_alpha_effect_is_detected_in_the_fixture(small_feature_table):
    """The confirmatory test must fire on data with a planted alpha effect."""
    res = st.occipital_alpha_summary(small_feature_table)
    assert res["mean_diff"] > 0
    assert res["dz"] > 0.8
    assert res["proportion_increased"] > 0.8


def test_participant_means_collapse_epochs_to_one_row_per_condition(small_feature_table):
    cols = fe.feature_columns(small_feature_table)[:5]
    pm = st.participant_means(small_feature_table, cols)
    n_subjects = small_feature_table["subject"].nunique()
    assert len(pm) == n_subjects * len(cfg.CONDITIONS)


# ------------------------------------------------------------------- leakage
def test_assert_no_leakage_raises_on_an_overlapping_split():
    with pytest.raises(AssertionError):
        tm.assert_no_leakage(np.array(["S001", "S002"]), np.array(["S002", "S003"]))


def test_assert_no_leakage_passes_on_a_clean_split():
    tm.assert_no_leakage(np.array(["S001", "S002"]), np.array(["S003"]))


def test_participant_split_shares_no_subject(small_feature_table):
    train, test = tm.split_by_participant(small_feature_table, test_size=0.34)
    assert not set(train["subject"]) & set(test["subject"])
    assert len(train) and len(test)


def test_participant_split_keeps_every_epoch_of_a_subject_together(small_feature_table):
    train, test = tm.split_by_participant(small_feature_table, test_size=0.34)
    for subject in small_feature_table["subject"].unique():
        in_train = subject in set(train["subject"])
        in_test = subject in set(test["subject"])
        assert in_train != in_test


def test_leaky_baseline_outperforms_the_honest_split(small_feature_table):
    """Quantifies the error the whole design exists to prevent.

    The epoch-level split lets a model recognise the participant and recall
    their condition.  It must score at least as well as the honest split; if it
    did not, the leakage argument in the paper would be unsupported.
    """
    train, test = tm.split_by_participant(small_feature_table, test_size=0.34)
    honest = tm.train_and_evaluate(train, test, models=["logistic_regression"],
                                   n_boot=0, verbose=False)
    leaky = tm.leaky_baseline(small_feature_table)
    assert leaky["roc_auc"] >= float(honest["roc_auc"].iloc[0]) - 1e-9


def test_scaler_is_fitted_inside_the_pipeline_not_on_full_data():
    """Every model must carry its own scaler, or folds leak into each other."""
    for name, model in tm.make_models().items():
        assert "scale" in dict(model.steps), f"{name} has no in-pipeline scaler"


# -------------------------------------------------------------------- metrics
def test_evaluate_reports_the_full_confusion_structure():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    res = tm.evaluate(y_true, y_pred, np.array([0.1, 0.6, 0.8, 0.9]))
    assert res["tn"] == 1 and res["fp"] == 1 and res["fn"] == 0 and res["tp"] == 2
    assert res["sensitivity"] == pytest.approx(1.0)
    assert res["specificity"] == pytest.approx(0.5)


def test_cluster_bootstrap_is_wider_than_a_naive_epoch_bootstrap():
    """Resampling epochs pretends dependent observations are independent.

    If the cluster interval were not wider, the paper's justification for using
    it would be empty.
    """
    rng = np.random.default_rng(3)
    groups = np.repeat([f"S{i:03d}" for i in range(12)], 40)
    y_true = np.tile([0, 1], len(groups) // 2)
    subject_skill = {g: rng.uniform(0.2, 0.95) for g in np.unique(groups)}
    y_score = np.array([
        rng.normal(subject_skill[g] if y else 1 - subject_skill[g], 0.25)
        for g, y in zip(groups, y_true)])
    y_pred = (y_score > 0.5).astype(int)

    clustered = tm.cluster_bootstrap_ci(y_true, y_score, y_pred, groups, n_boot=400)
    naive = tm.cluster_bootstrap_ci(y_true, y_score, y_pred,
                                    np.arange(len(groups)), n_boot=400)
    clustered_width = clustered["roc_auc_ci_high"] - clustered["roc_auc_ci_low"]
    naive_width = naive["roc_auc_ci_high"] - naive["roc_auc_ci_low"]
    assert clustered_width > naive_width


# ------------------------------------------------------------- reproducibility
def test_split_is_deterministic_under_a_fixed_seed(small_feature_table):
    a, _ = tm.split_by_participant(small_feature_table, seed=cfg.RANDOM_SEED)
    b, _ = tm.split_by_participant(small_feature_table, seed=cfg.RANDOM_SEED)
    assert sorted(a["subject"].unique()) == sorted(b["subject"].unique())


def test_different_seeds_give_different_splits(small_feature_table):
    a, _ = tm.split_by_participant(small_feature_table, seed=1)
    b, _ = tm.split_by_participant(small_feature_table, seed=99)
    assert len(small_feature_table["subject"].unique()) >= 4


def test_training_is_deterministic_under_a_fixed_seed(small_feature_table):
    train, test = tm.split_by_participant(small_feature_table, test_size=0.34)
    cols = fe.feature_columns(train)[:60]
    first = tm.train_and_evaluate(train, test, columns=cols,
                                  models=["logistic_regression"],
                                  n_boot=0, verbose=False)
    second = tm.train_and_evaluate(train, test, columns=cols,
                                   models=["logistic_regression"],
                                   n_boot=0, verbose=False)
    assert first["roc_auc"].iloc[0] == pytest.approx(second["roc_auc"].iloc[0])


def test_permuted_labels_score_at_chance(small_feature_table):
    """Negative control.

    Labels shuffled WITHIN each participant destroy the condition signal while
    preserving participant identity.  A model that still scores well is reading
    something it should not.
    """
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    df = small_feature_table.copy()
    df["label"] = (df.groupby("subject")["label"]
                   .transform(lambda s: rng.permutation(s.to_numpy())))
    train, test = tm.split_by_participant(df, test_size=0.34)
    cols = fe.feature_columns(train)[:80]
    res = tm.train_and_evaluate(train, test, columns=cols,
                                models=["logistic_regression"],
                                n_boot=0, verbose=False)
    assert float(res["roc_auc"].iloc[0]) == pytest.approx(0.5, abs=0.20)
