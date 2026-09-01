"""Statistical analysis at the participant level.

The unit of analysis is the participant, not the epoch.  Each participant
contributes roughly fifty epochs per condition, and those epochs are not
independent observations -- they share a skull, an electrode placement, and an
alpha peak frequency.  Testing at the epoch level would inflate the effective
sample size roughly fiftyfold and produce p-values that mean nothing.  So every
test here first averages each participant's epochs within condition, then
compares the two conditions across participants.

Because every participant provides both conditions, the design is paired.  That
is a real advantage: it removes between-participant variance, which in EEG is
large, and it makes the effect size a within-subject Cohen's dz.

Compositional caution
---------------------
Relative band power is compositional: the five bands sum to one, so a genuine
rise in one band mechanically depresses every other.  Any conclusion drawn from
relative power alone is therefore suspect, and every test below is repeated on
log absolute power so the two can be compared.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402


# ---------------------------------------------------------------- primitives
def cohens_dz(diff: np.ndarray) -> float:
    """Within-subject effect size for paired differences."""
    diff = np.asarray(diff, dtype=float)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def paired_ci(diff: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Student-t confidence interval on the mean paired difference."""
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    if n < 2:
        return (np.nan, np.nan)
    se = diff.std(ddof=1) / np.sqrt(n)
    crit = stats.t.ppf(1 - alpha / 2, n - 1)
    return float(diff.mean() - crit * se), float(diff.mean() + crit * se)


def benjamini_hochberg(pvals: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rejected, adjusted_p)`` under Benjamini-Hochberg FDR control.

    Bonferroni over 320 channel-by-band comparisons would be so conservative
    that only enormous effects survive.  Controlling the false discovery rate
    instead accepts that a fixed proportion of the rejections will be false,
    which is the appropriate trade-off for an exploratory topographic map.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out <= q, out


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired t-test with a Wilcoxon signed-rank check on the same data.

    The t-test assumes the paired differences are roughly normal.  Wilcoxon
    does not, so agreement between the two is evidence that the assumption is
    not doing the work.  Disagreement is reported rather than resolved.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = b - a
    n = len(diff)
    res = {"n": n, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
           "mean_diff": float(diff.mean()), "dz": cohens_dz(diff)}
    res["ci_low"], res["ci_high"] = paired_ci(diff)
    if n >= 2 and diff.std(ddof=1) > 0:
        t, p = stats.ttest_rel(b, a)
        res["t"], res["p_ttest"] = float(t), float(p)
        try:
            w, pw = stats.wilcoxon(b, a)
            res["wilcoxon_W"], res["p_wilcoxon"] = float(w), float(pw)
        except ValueError:
            res["wilcoxon_W"], res["p_wilcoxon"] = np.nan, np.nan
    else:
        res.update({"t": np.nan, "p_ttest": np.nan,
                    "wilcoxon_W": np.nan, "p_wilcoxon": np.nan})
    res["n_positive"] = int((diff > 0).sum())
    return res


# --------------------------------------------------------------- aggregation
def participant_means(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Average each participant's epochs within condition."""
    return (features.groupby(["subject", "condition"], observed=True)[columns]
            .mean().reset_index())


def _paired_arrays(pm: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    wide = pm.pivot(index="subject", columns="condition", values=column).dropna()
    return wide["eyes_open"].to_numpy(), wide["eyes_closed"].to_numpy()


# ------------------------------------------------------------------ analyses
def channel_band_table(features: pd.DataFrame, prefix: str = "rel",
                       q: float = 0.05) -> pd.DataFrame:
    """Paired test for every channel-by-band feature with the given prefix."""
    cols = [c for c in features.columns if c.startswith(f"{prefix}_")]
    pm = participant_means(features, cols)
    rows = []
    for col in cols:
        _, ch, band = col.split("_", 2)
        open_, closed = _paired_arrays(pm, col)
        rows.append({"feature": col, "channel": ch, "band": band,
                     **paired_test(open_, closed)})
    df = pd.DataFrame(rows)
    if len(df):
        rejected, adj = benjamini_hochberg(df["p_ttest"].to_numpy(), q)
        df["p_fdr"] = adj
        df["significant"] = rejected
    return df.sort_values("dz", key=np.abs, ascending=False).reset_index(drop=True)


def global_band_table(features: pd.DataFrame) -> pd.DataFrame:
    """Whole-scalp relative power against occipital log absolute power.

    This is the comparison that separates a genuine spectral change from a
    compositional artifact.  A band whose relative power moves but whose
    absolute power does not has not actually changed; it has been renormalised
    by a change somewhere else in the spectrum.
    """
    occ = [c for c in cfg.OCCIPITAL]
    rows = []
    for band in cfg.BAND_NAMES:
        rel_col = f"global_{band}"
        abs_cols = [f"logabs_{ch}_{band}" for ch in occ
                    if f"logabs_{ch}_{band}" in features.columns]
        if rel_col not in features.columns or not abs_cols:
            continue
        tmp = features[["subject", "condition"]].copy()
        tmp["rel"] = features[rel_col]
        tmp["abs"] = features[abs_cols].mean(axis=1)
        pm = participant_means(tmp, ["rel", "abs"])

        ro, rc = _paired_arrays(pm, "rel")
        ao, ac = _paired_arrays(pm, "abs")
        r, a = paired_test(ro, rc), paired_test(ao, ac)
        rows.append({
            "band": band,
            "rel_diff": r["mean_diff"], "rel_dz": r["dz"], "rel_p": r["p_ttest"],
            "abs_diff": a["mean_diff"], "abs_dz": a["dz"], "abs_p": a["p_ttest"],
            "abs_fold_change": 10 ** a["mean_diff"],
            "n_positive_abs": a["n_positive"], "n": a["n"],
        })
    df = pd.DataFrame(rows)
    if len(df):
        for kind in ("rel", "abs"):
            _, adj = benjamini_hochberg(df[f"{kind}_p"].to_numpy())
            df[f"{kind}_q"] = adj
        df["interpretation"] = [
            _interpret(row) for _, row in df.iterrows()
        ]
    return df


def _interpret(row: pd.Series) -> str:
    rel_sig = row["rel_q"] < 0.05
    abs_sig = row["abs_q"] < 0.05
    if abs_sig and abs(row["abs_dz"]) >= 1.0:
        return "genuine, large"
    if abs_sig and rel_sig and np.sign(row["abs_dz"]) != np.sign(row["rel_dz"]):
        return "genuine change, masked by relative power"
    if abs_sig:
        return "genuine change"
    if rel_sig:
        return "compositional artifact"
    return "no reliable change"


def occipital_alpha_summary(features: pd.DataFrame) -> dict:
    """The confirmatory test: does occipital relative alpha rise on eye closure?

    If this fails, the whole cost analysis is meaningless, so it is checked
    explicitly rather than assumed.
    """
    cols = [f"rel_{ch}_alpha" for ch in cfg.OCCIPITAL if f"rel_{ch}_alpha" in features.columns]
    tmp = features[["subject", "condition"]].copy()
    tmp["occ_rel_alpha"] = features[cols].mean(axis=1)
    pm = participant_means(tmp, ["occ_rel_alpha"])
    open_, closed = _paired_arrays(pm, "occ_rel_alpha")
    res = paired_test(open_, closed)
    res["n_increased"] = res.pop("n_positive")
    res["proportion_increased"] = res["n_increased"] / res["n"] if res["n"] else 0.0
    return res


def rejection_balance(qc: pd.DataFrame) -> dict:
    """Is epoch rejection balanced across conditions?

    Differential data loss between conditions is itself a confound, so this is
    tested rather than assumed.  A significant imbalance is reported, not
    corrected away -- the robustness analysis then checks whether the result
    survives with rejection disabled entirely.
    """
    wide = qc.pivot_table(index="subject", columns="condition",
                          values="reject_fraction")
    wide = wide.dropna()
    if wide.empty or not {"eyes_open", "eyes_closed"} <= set(wide.columns):
        return {}
    o, c = wide["eyes_open"].to_numpy(), wide["eyes_closed"].to_numpy()
    out = paired_test(o, c)
    out["mean_reject_eyes_open"] = float(o.mean())
    out["mean_reject_eyes_closed"] = float(c.mean())
    return out


def main() -> int:
    _, results_dir = cfg.output_dirs()
    features = pd.read_csv(cfg.FEATURES_CSV)

    summary = occipital_alpha_summary(features)
    print("Occipital relative alpha, eyes closed minus eyes open")
    print(f"  n                   {summary['n']}")
    print(f"  increased in        {summary['n_increased']}/{summary['n']} participants")
    print(f"  mean difference     {summary['mean_diff']:+.4f} "
          f"[{summary['ci_low']:+.4f}, {summary['ci_high']:+.4f}]")
    print(f"  Cohen's dz          {summary['dz']:+.3f}")
    print(f"  t                   {summary['t']:.3f}   p = {summary['p_ttest']:.3g}")
    print(f"  Wilcoxon            p = {summary['p_wilcoxon']:.3g}")
    pd.DataFrame([summary]).to_csv(results_dir / "occipital_alpha_summary.csv", index=False)

    gb = global_band_table(features)
    gb.to_csv(results_dir / "band_comparison.csv", index=False)
    print("\nWhole-scalp relative vs occipital absolute power")
    print(gb[["band", "rel_diff", "rel_dz", "abs_diff", "abs_dz",
              "abs_fold_change", "interpretation"]].to_string(index=False))

    for prefix in ("rel", "logabs"):
        tab = channel_band_table(features, prefix)
        tab.to_csv(results_dir / f"channel_band_tests_{prefix}.csv", index=False)
        print(f"\n{prefix}: {int(tab['significant'].sum())}/{len(tab)} "
              f"channel x band comparisons survive FDR")
        print(tab.head(5)[["feature", "mean_diff", "dz", "p_fdr"]].to_string(index=False))

    if cfg.QC_CSV.exists():
        bal = rejection_balance(pd.read_csv(cfg.QC_CSV))
        if bal:
            pd.DataFrame([bal]).to_csv(results_dir / "rejection_balance.csv", index=False)
            print(f"\nEpoch rejection: eyes open {bal['mean_reject_eyes_open']:.3f} "
                  f"vs eyes closed {bal['mean_reject_eyes_closed']:.3f}, "
                  f"Wilcoxon p = {bal['p_wilcoxon']:.3g}")
    print(f"\nresults -> {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
