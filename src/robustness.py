"""Robustness analysis: does the conclusion survive the choices I made?

Every preprocessing decision in this project could have gone another way.  The
honest test of a conclusion is not whether it holds under the settings that
produced it, but whether it holds under the settings that did not.

Five families of variation are tested:

  * artifact-rejection threshold, including disabling rejection entirely --
    this is the direct check on the differential-rejection confound;
  * feature family, including removing every alpha feature, which asks whether
    the result depends on the rhythm the whole literature attributes it to;
  * referencing, with and without the average reference;
  * band definitions, standard against a shifted alpha edge;
  * ICA, applied and not applied.

Re-extraction is required for the rejection, reference, and ICA variants
because they change the epochs themselves.  Feature-family and band variants
can reuse the cached matrix, so they are cheap.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import complexity_analysis as ca  # noqa: E402
from src import config as cfg  # noqa: E402
from src import train_models as tm  # noqa: E402
from src.feature_extraction import build_feature_matrix, feature_columns  # noqa: E402


def _score_matrix(feats: pd.DataFrame, columns: list[str] | None = None) -> dict:
    train, test = tm.split_by_participant(feats)
    return ca.score_columns(train, test, columns or feature_columns(feats))


# ------------------------------------------------------- cheap variants
def feature_family_variants(df: pd.DataFrame) -> list[dict]:
    """Which feature families are load-bearing?

    The row that matters is ``exclude_alpha``.  If removing every alpha feature
    barely changes performance, the information is redundant across bands --
    alpha is sufficient for classification but not necessary, because eye
    closure reshapes the entire spectrum rather than one rhythm in isolation.
    """
    cols = feature_columns(df)
    families = {
        "all_features": cols,
        "relative_only": [c for c in cols if c.startswith("rel_")],
        "absolute_only": [c for c in cols if c.startswith("logabs_")],
        "alpha_only": [c for c in cols if c.endswith("_alpha")],
        "exclude_alpha": [c for c in cols if not c.endswith("_alpha")],
    }
    rows = []
    for name, subset in families.items():
        if not subset:
            continue
        rows.append({"analysis": "feature_family", "setting": name,
                     **_score_matrix(df, subset)})
    return rows


def band_definition_variants(subjects: list[str] | None) -> list[dict]:
    """Standard band edges against a shifted alpha/theta boundary."""
    rows = []
    for name, bands in (("standard", cfg.BANDS), ("alt_alpha_7_13", cfg.BANDS_ALT)):
        feats, _ = build_feature_matrix(subjects, bands=bands, verbose=False)
        rows.append({"analysis": "band_definition", "setting": name,
                     **_score_matrix(feats)})
    return rows


# ------------------------------------------------- variants needing re-extraction
def rejection_variants(subjects: list[str] | None) -> list[dict]:
    """Sweep the artifact-rejection threshold, including no rejection at all.

    If performance is unchanged with rejection disabled, then the differential
    rejection between conditions -- more eyes-open epochs discarded, consistent
    with blinks -- cannot be driving any result.
    """
    rows = []
    for ptp in cfg.REJECT_PTP_SWEEP:
        label = "none" if ptp is None else f"{ptp:.0f}uV"
        feats, _ = build_feature_matrix(subjects, ptp_uv=ptp, verbose=False)
        rows.append({"analysis": "artifact_rejection", "setting": label,
                     "n_subjects": feats["subject"].nunique(),
                     "n_epochs": len(feats), **_score_matrix(feats)})
    return rows


def reference_variants(subjects: list[str] | None) -> list[dict]:
    """With and without the average reference.

    In the published run this was the only preprocessing choice that mattered
    materially, which is worth knowing: an average reference attenuates
    spatially broad activity, and removing it costs real performance.
    """
    rows = []
    for name, avg in (("average", True), ("none", False)):
        feats, _ = build_feature_matrix(subjects, average_reference=avg, verbose=False)
        rows.append({"analysis": "referencing", "setting": name,
                     **_score_matrix(feats)})
    return rows


def ica_variants(subjects: list[str] | None) -> list[dict]:
    """ICA applied and not applied.

    ICA is off in the main pipeline because these recordings have no EOG
    channel, so ocular components must be identified heuristically and a
    misidentification removes genuine frontal activity.  This checks whether
    that omission cost anything.
    """
    rows = []
    for name, use in (("not_applied", False), ("applied_fp1_proxy", True)):
        feats, _ = build_feature_matrix(subjects, apply_ica=use, verbose=False)
        rows.append({"analysis": "ica", "setting": name, **_score_matrix(feats)})
    return rows


# ---------------------------------------------------------------------- main
def run_all(subjects: list[str] | None = None,
            include_expensive: bool = True) -> pd.DataFrame:
    df = tm.load_features()
    rows: list[dict] = []

    print("  feature families")
    rows += feature_family_variants(df)
    if include_expensive:
        print("  artifact rejection")
        rows += rejection_variants(subjects)
        print("  referencing")
        rows += reference_variants(subjects)
        print("  band definitions")
        rows += band_definition_variants(subjects)
        print("  ICA")
        rows += ica_variants(subjects)

    out = pd.DataFrame(rows)
    cols = ["analysis", "setting", "roc_auc", "accuracy", "n_features"]
    return out[[c for c in cols if c in out.columns] +
               [c for c in out.columns if c not in cols]]


def main() -> int:
    _, results_dir = cfg.output_dirs()
    table = run_all()
    table.to_csv(results_dir / "robustness.csv", index=False)
    print("\nRobustness across preprocessing and feature choices:")
    print(table[["analysis", "setting", "roc_auc", "accuracy"]].to_string(index=False))
    span = table["roc_auc"].max() - table["roc_auc"].min()
    print(f"\nAUC ranges {table['roc_auc'].min():.3f} to {table['roc_auc'].max():.3f} "
          f"across {len(table)} configurations (span {span:.3f})")
    print(f"results -> {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
