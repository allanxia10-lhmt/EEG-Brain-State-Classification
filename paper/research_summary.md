# One-page research summary

## Research question

Can EEG distinguish eyes-open from eyes-closed brain states — and how few
electrodes, how little recording time, and how simple a model does it take?

The first half has been settled since 1929. The contribution is the second
half: measuring the *marginal value* of each resource rather than reporting one
more accuracy figure.

## Hypotheses

- **H1** A single occipital relative-alpha feature will come within a small
  margin of a full multivariate model, because the effect is large and
  low-dimensional.
- **H2** Performance will saturate at a small number of electrodes, with
  placement mattering more than count at small budgets.
- **H3** Window length will trade off against electrode count, since both
  increase the evidence available per decision.

## Dataset

PhysioNet EEG Motor Movement/Imagery Dataset (eegmmidb) v1.0.0. 109 adults, 64
channels, 160 Hz, ODC-BY licensed. Only the two 61-second baseline runs are
used: R01 (eyes open) and R02 (eyes closed). 106 participants provided usable
data in both conditions; three lost an entire condition to artifact rejection
and were excluded, since a paired design requires both.

## Methods

1–45 Hz zero-phase FIR band-pass (45 Hz sits below the 60 Hz mains line, so no
notch is needed), correlation-based bad-channel detection and interpolation,
average reference, 2-second non-overlapping epochs, and fraction-based artifact
rejection. 674 spectral features per epoch from Welch PSD: log absolute and
relative band power per channel, regional and global summaries, and four
derived features including individual alpha peak frequency.

All partitions are over **participants**, never epochs. Confidence intervals
come from a cluster bootstrap that resamples participants.

## Key results

| Claim | Number |
|---|---|
| Occipital relative alpha increased on eye closure | **106 / 106** participants, dz = **+1.59** |
| One occipital electrode, one alpha feature | ROC-AUC **0.829** |
| 64 electrodes, 674 features, tuned model | ROC-AUC **0.894** |
| Best nonlinear model | 0.892 — no advantage over logistic regression |
| 64 → 19 electrodes | costs **0.004** AUC |
| Placement advantage at 1 electrode / at 9 | **+0.084** / **~0** |
| 1 electrode @ 8 s vs 64 electrodes @ 0.5 s | **0.856** vs **0.857** — interchangeable |
| Removing *every* alpha feature | costs **0.004** AUC |
| Epoch-level split instead of participant-level | inflates AUC 0.894 → **0.940** |
| LOSO across 106 participants | 0.778 ± 0.155; **17** participants below 0.60 |

Across 15 preprocessing configurations AUC spanned 0.850–0.898. The only choice
that mattered was re-referencing.

## Important figures

- `06_complexity_tradeoff.png` — the primary result: ablation, window curves,
  and the electrode × time trade-off surface.
- `07b_feature_ladder.png` — what feature richness actually buys, against the
  inflation caused by leaky cross-validation.
- `04_band_power_comparison.png` — all 106 participants, paired, plus the
  absolute-power effect sizes that separate genuine change from renormalisation.
- `11_subject_level_performance.png` — the long tail the mean conceals.

## Conclusion

Performance in this paradigm is constrained by physiology and recording
duration, not by feature engineering or model capacity. For a low-cost system
targeting this class of contrast, spend the budget on **electrode placement**
and **window length**, not on channel count or model sophistication; validate on
held-out participants; and report the per-participant distribution rather than
the mean.

Alpha is *sufficient but not necessary* for classification: it carries the
largest univariate effect, yet removing it costs almost nothing, because eye
closure reshapes the whole spectrum rather than one rhythm in isolation.

## Limitations

Condition is confounded with run (separate consecutive recordings). Epoch
rejection is differential across conditions, though results survive disabling it
entirely. Recordings are only 61 s, so the window-length curve had not
saturated. Electrode subsets are *simulated* by discarding channels from a
64-channel average-referenced montage — a real one-electrode system would have a
different reference and no neighbours to interpolate from. Most importantly:
these cost curves describe one large, spatially focal effect. A subtler contrast
could easily behave differently, and complexity might earn its keep there.

## Future work

Test whether the saturation shape holds for harder contrasts where the effect is
smaller and more distributed; whether personalised or few-shot calibration
rescues the poorly classified tail; whether these curves survive on genuinely
low-channel dry-electrode hardware rather than simulated subsets; and whether
the beta increase reported here is an alpha harmonic or physiologically
independent.
