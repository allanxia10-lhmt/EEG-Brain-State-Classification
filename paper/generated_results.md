# Generated results

Every number below was produced by the pipeline, not typed by hand. Regenerate with `python src/report.py`.

- data source: `['cache']`
- checksums verified: `True`
- synthetic fixture: `False`
- retrieved: `2026-09-01T17:02:08+00:00`

## Sample

- participants: **106**
- epochs: **5704** (eyes_closed 2959, eyes_open 2745)
- features per epoch: **674**

## Confirmatory test: occipital relative alpha

- increased in **106/106** participants
- mean paired difference **+0.292** [+0.257, +0.327]
- Cohen's dz **+1.60**, t(105) = 16.45, p = 8.11e-31, Wilcoxon p = 3.99e-19

## Band comparison: relative vs absolute power

| band   |   rel_diff |   rel_dz |    rel_q |   abs_diff |   abs_dz |    abs_q |   abs_fold_change | interpretation                           |
|:-------|-----------:|---------:|---------:|-----------:|---------:|---------:|------------------:|:-----------------------------------------|
| delta  |    -0.112  |   -1.13  | 3.53e-20 |    -0.149  |   -0.538 | 3.81e-07 |              0.71 | genuine change                           |
| theta  |    -0.0191 |   -0.485 | 3.02e-06 |     0.0417 |    0.172 | 0.0792   |              1.1  | compositional artifact                   |
| alpha  |     0.17   |    1.41  | 3.79e-26 |     0.735  |    1.65  | 3.48e-31 |              5.43 | genuine, large                           |
| beta   |    -0.0146 |   -0.241 | 0.0145   |     0.231  |    1.21  | 4.82e-22 |              1.7  | genuine, large                           |
| gamma  |    -0.0246 |   -0.677 | 4.72e-10 |     0.0254 |    0.218 | 0.0335   |              1.06 | genuine change, masked by relative power |

## Feature ladder

| feature_set             |   n_features |   accuracy |   accuracy_ci_low |   accuracy_ci_high |   roc_auc |   roc_auc_ci_low |   roc_auc_ci_high |    f1 |   sensitivity |   specificity |
|:------------------------|-------------:|-----------:|------------------:|-------------------:|----------:|-----------------:|------------------:|------:|--------------:|--------------:|
| dummy_baseline          |            0 |      0.514 |                   |                    |     0.5   |                  |                   | 0.679 |         1     |         0     |
| single_occipital_alpha  |            1 |      0.751 |             0.693 |              0.811 |     0.83  |            0.776 |             0.883 | 0.717 |         0.614 |         0.896 |
| whole_scalp_band_powers |            5 |      0.691 |             0.632 |              0.752 |     0.777 |            0.714 |             0.841 | 0.681 |         0.641 |         0.743 |
| per_channel_relative    |          320 |      0.799 |             0.745 |              0.853 |     0.883 |            0.834 |             0.929 | 0.786 |         0.718 |         0.885 |
| all_features            |          674 |      0.801 |             0.749 |              0.853 |     0.894 |            0.849 |             0.937 | 0.794 |         0.746 |         0.859 |

## Model comparison

| model                  |   n_features |   accuracy |   roc_auc |   precision |   recall |    f1 | best_params                                                                    |
|:-----------------------|-------------:|-----------:|----------:|------------:|---------:|------:|:-------------------------------------------------------------------------------|
| dummy                  |          674 |      0.514 |     0.5   |       0.514 |    1     | 0.679 | {}                                                                             |
| logistic_regression    |          674 |      0.793 |     0.893 |       0.862 |    0.711 | 0.78  | {'clf__C': 0.001}                                                              |
| linear_svm             |          674 |      0.798 |     0.891 |       0.847 |    0.742 | 0.791 | {'clf__C': 0.001}                                                              |
| random_forest          |          674 |      0.775 |     0.876 |       0.866 |    0.666 | 0.753 | {'clf__max_depth': None, 'clf__min_samples_leaf': 4, 'clf__n_estimators': 300} |
| hist_gradient_boosting |          674 |      0.791 |     0.897 |       0.854 |    0.716 | 0.779 | {'clf__learning_rate': 0.05, 'clf__max_iter': 200, 'clf__max_leaf_nodes': 15}  |

## Electrode ablation

| electrode_set   |   n_electrodes |   roc_auc |   accuracy |   random_auc_mean |   random_auc_min |   random_auc_max |   placement_advantage |
|:----------------|---------------:|----------:|-----------:|------------------:|-----------------:|-----------------:|----------------------:|
| all_64          |             64 |     0.894 |      0.801 |                   |                  |                  |                       |
| ten_twenty_19   |             19 |     0.889 |      0.781 |             0.868 |            0.846 |            0.888 |               0.0206  |
| occipital_9     |              9 |     0.844 |      0.754 |             0.847 |            0.82  |            0.864 |              -0.00368 |
| occipital_3     |              3 |     0.83  |      0.742 |             0.806 |            0.728 |            0.862 |               0.024   |
| occipital_1     |              1 |     0.817 |      0.741 |             0.763 |            0.685 |            0.824 |               0.0537  |
| headband_3      |              3 |     0.776 |      0.684 |             0.791 |            0.766 |            0.843 |              -0.0146  |
| headband_1      |              1 |     0.739 |      0.663 |             0.755 |            0.697 |            0.824 |              -0.0159  |

## Window length

|   window_s |   n_subjects |   n_test_epochs |   accuracy |   roc_auc |   roc_auc_ci_low |   roc_auc_ci_high |
|-----------:|-------------:|----------------:|-----------:|----------:|-----------------:|------------------:|
|        0.5 |          109 |            7865 |      0.772 |     0.864 |            0.822 |             0.903 |
|        1   |          109 |            3816 |      0.789 |     0.879 |            0.839 |             0.92  |
|        2   |          106 |            1798 |      0.801 |     0.894 |            0.851 |             0.94  |
|        4   |           99 |             774 |      0.86  |     0.938 |            0.899 |             0.971 |
|        8   |           84 |             303 |      0.822 |     0.917 |            0.868 |             0.963 |

## Robustness

| analysis           | setting           |   roc_auc |   accuracy |
|:-------------------|:------------------|----------:|-----------:|
| feature_family     | all_features      |     0.894 |      0.801 |
| feature_family     | relative_only     |     0.883 |      0.799 |
| feature_family     | absolute_only     |     0.9   |      0.805 |
| feature_family     | alpha_only        |     0.873 |      0.781 |
| feature_family     | exclude_alpha     |     0.892 |      0.8   |
| artifact_rejection | none              |     0.894 |      0.801 |
| artifact_rejection | 200uV             |     0.889 |      0.812 |
| artifact_rejection | 250uV             |     0.894 |      0.801 |
| artifact_rejection | 350uV             |     0.911 |      0.809 |
| referencing        | average           |     0.894 |      0.801 |
| referencing        | none              |     0.869 |      0.781 |
| band_definition    | standard          |     0.894 |      0.801 |
| band_definition    | alt_alpha_7_13    |     0.893 |      0.803 |
| ica                | not_applied       |     0.894 |      0.801 |
| ica                | applied_fp1_proxy |     0.867 |      0.777 |

## Permutation importance by band

| band    |   importance_mean |
|:--------|------------------:|
| beta    |           0.0368  |
| gamma   |           0.0289  |
| alpha   |           0.0174  |
| delta   |           0.00803 |
| theta   |           0.00698 |
| derived |           0.00246 |

## Electrode x window trade-off surface (ROC-AUC)

| electrode_set   |   0.5 |   1.0 |   2.0 |   4.0 |   8.0 |
|:----------------|------:|------:|------:|------:|------:|
| all_64          | 0.864 | 0.879 | 0.894 | 0.938 | 0.917 |
| ten_twenty_19   | 0.824 | 0.843 | 0.889 | 0.928 | 0.902 |
| occipital_9     | 0.813 | 0.848 | 0.844 | 0.899 | 0.887 |
| occipital_3     | 0.792 | 0.829 | 0.83  | 0.89  | 0.888 |
| occipital_1     | 0.771 | 0.814 | 0.817 | 0.886 | 0.878 |
| headband_3      | 0.768 | 0.804 | 0.776 | 0.854 | 0.881 |
| headband_1      | 0.732 | 0.774 | 0.739 | 0.85  | 0.875 |

## Leave-one-subject-out

- accuracy **0.777 +/- 0.160**, ROC-AUC **0.877**
- median 0.827, minimum 0.240
- **18/106** participants below 0.60

## Leakage demonstration

| split | accuracy | ROC-AUC |
|---|---|---|
| participant-level (correct) | 0.793 | 0.893 |
| epoch-level (wrong) | 0.852 | 0.940 |
| **inflation** | **+0.059** | **+0.048** |
