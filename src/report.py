"""Regenerate every numeric claim in the paper from the results on disk.

The rule this project runs on is that no number in the manuscript is typed by
hand.  This script reads results/*.csv and writes paper/generated_results.md,
so the manuscript can be checked against an actual run rather than against
memory.  If a number in the paper is not in this file, it did not come from the
pipeline and should not be in the paper.

    python src/report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402


def _read(d: Path, name: str) -> pd.DataFrame | None:
    p = d / name
    return pd.read_csv(p) if p.exists() else None


def _md(df: pd.DataFrame, cols: list[str] | None = None, nd: int = 3) -> str:
    if df is None or df.empty:
        return "_(not available -- run the pipeline first)_\n"
    sub = df[[c for c in (cols or df.columns) if c in df.columns]].copy()
    for c in sub.columns:
        if pd.api.types.is_float_dtype(sub[c]):
            sub[c] = sub[c].map(lambda v: f"{v:.{nd}g}" if pd.notna(v) else "")
    return sub.to_markdown(index=False) + "\n"


def build() -> str:
    _, res = cfg.output_dirs()
    prov = cfg.read_provenance()
    out: list[str] = []
    add = out.append

    add("# Generated results\n")
    add("Every number below was produced by the pipeline, not typed by hand. "
        "Regenerate with `python src/report.py`.\n")
    add(f"- data source: `{prov.get('source')}`")
    add(f"- checksums verified: `{prov.get('verified')}`")
    add(f"- synthetic fixture: `{prov.get('synthetic')}`")
    add(f"- retrieved: `{prov.get('retrieved_utc', 'unknown')}`\n")
    if prov.get("synthetic"):
        add("> **These are smoke-test numbers from simulated data. "
            "They are not results and must never appear in the paper.**\n")

    if cfg.FEATURES_CSV.exists():
        f = pd.read_csv(cfg.FEATURES_CSV)
        from src.feature_extraction import feature_columns
        add("## Sample\n")
        add(f"- participants: **{f['subject'].nunique()}**")
        add(f"- epochs: **{len(f)}** "
            f"({', '.join(f'{k} {v}' for k, v in f.groupby('condition').size().items())})")
        add(f"- features per epoch: **{len(feature_columns(f))}**\n")

    alpha = _read(res, "occipital_alpha_summary.csv")
    if alpha is not None:
        r = alpha.iloc[0]
        add("## Confirmatory test: occipital relative alpha\n")
        add(f"- increased in **{int(r['n_increased'])}/{int(r['n'])}** participants")
        add(f"- mean paired difference **{r['mean_diff']:+.3f}** "
            f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")
        add(f"- Cohen's dz **{r['dz']:+.2f}**, t({int(r['n'])-1}) = {r['t']:.2f}, "
            f"p = {r['p_ttest']:.3g}, Wilcoxon p = {r['p_wilcoxon']:.3g}\n")

    for title, name, cols in [
        ("Band comparison: relative vs absolute power", "band_comparison.csv",
         ["band", "rel_diff", "rel_dz", "rel_q", "abs_diff", "abs_dz", "abs_q",
          "abs_fold_change", "interpretation"]),
        ("Feature ladder", "feature_ladder.csv",
         ["feature_set", "n_features", "accuracy", "accuracy_ci_low",
          "accuracy_ci_high", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high",
          "f1", "sensitivity", "specificity"]),
        ("Model comparison", "model_comparison.csv",
         ["model", "n_features", "accuracy", "roc_auc", "precision", "recall",
          "f1", "best_params"]),
        ("Electrode ablation", "electrode_ablation.csv",
         ["electrode_set", "n_electrodes", "roc_auc", "accuracy",
          "random_auc_mean", "random_auc_min", "random_auc_max",
          "placement_advantage"]),
        ("Window length", "window_sweep.csv",
         ["window_s", "n_subjects", "n_test_epochs", "accuracy", "roc_auc",
          "roc_auc_ci_low", "roc_auc_ci_high"]),
        ("Robustness", "robustness.csv", ["analysis", "setting", "roc_auc", "accuracy"]),
        ("Permutation importance by band", "feature_importance_by_band.csv", None),
    ]:
        df = _read(res, name)
        add(f"## {title}\n")
        add(_md(df, cols))

    surface = _read(res, "tradeoff_surface.csv")
    if surface is not None:
        add("## Electrode x window trade-off surface (ROC-AUC)\n")
        pivot = surface.pivot(index="electrode_set", columns="window_s",
                              values="roc_auc")
        order = [k for k in cfg.ELECTRODE_SETS if k in pivot.index]
        add(pivot.reindex(order).round(3).to_markdown() + "\n")

    loso = _read(res, "loso_summary.csv")
    if loso is not None:
        r = loso.iloc[0]
        add("## Leave-one-subject-out\n")
        add(f"- accuracy **{r['mean_subject_accuracy']:.3f} "
            f"+/- {r['std_subject_accuracy']:.3f}**, ROC-AUC **{r['roc_auc']:.3f}**")
        add(f"- median {r['median_subject_accuracy']:.3f}, "
            f"minimum {r['min_subject_accuracy']:.3f}")
        add(f"- **{int(r['n_subjects_below_0.60'])}/{int(r['n_subjects'])}** "
            f"participants below 0.60\n")

    leak = _read(res, "leakage_demonstration.csv")
    models = _read(res, "model_comparison.csv")
    if leak is not None and models is not None:
        honest = models.set_index("model").loc["logistic_regression"]
        r = leak.iloc[0]
        add("## Leakage demonstration\n")
        add("| split | accuracy | ROC-AUC |")
        add("|---|---|---|")
        add(f"| participant-level (correct) | {honest['accuracy']:.3f} | "
            f"{honest['roc_auc']:.3f} |")
        add(f"| epoch-level (wrong) | {r['accuracy']:.3f} | {r['roc_auc']:.3f} |")
        add(f"| **inflation** | **{r['accuracy'] - honest['accuracy']:+.3f}** | "
            f"**{r['roc_auc'] - honest['roc_auc']:+.3f}** |\n")

    return "\n".join(out)


def main() -> int:
    text = build()
    path = cfg.PAPER / "generated_results.md"
    path.write_text(text)
    print(f"paper tables -> {path}")
    print(f"  {len(text.splitlines())} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
