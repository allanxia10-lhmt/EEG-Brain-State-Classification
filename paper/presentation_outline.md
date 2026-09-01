# Presentation outline — 10 slides with speaker notes

Target: 12 minutes plus questions. The arc is deliberately anti-climactic in
one specific way — the "result" everyone expects arrives on slide 7 and is then
immediately deflated. That is the point of the talk.

---

## 1. Title

**Diminishing returns of electrode count, epoch duration, and model complexity
in EEG brain-state classification**

> Speaker notes: Say the one-line version immediately, before any background:
> "You can tell eyes open from eyes closed with one electrode and one number.
> Everything past that buys you about seven points of AUC." Then promise that
> the interesting part is the *shape* of the curve, not the endpoint.

## 2. Motivation

A consumer EEG headband costs about $200. A research system costs $50,000 and
uses 64 gel electrodes. Which do you actually need?

> Speaker notes: Frame it as a purchasing decision, not a science question — it
> makes the whole talk concrete. Most EEG-ML papers report a final accuracy and
> never tell you what the apparatus contributed. Say plainly that you find that
> unsatisfying, and that this project is the answer to "compared to what?"

## 3. Neuroscience background

Alpha (8–13 Hz), posterior, large with eyes closed, attenuates on eye opening.
The modern account is not idling but **active inhibition**: alpha as pulsed
suppression of visual cortex under thalamocortical control.

> Speaker notes: Two minutes maximum. The audience does not need the full
> gating-by-inhibition literature. What they need is: the effect is real, it is
> enormous, and it is spatially focal at the back of the head. That last point
> is what makes the electrode-placement result interpretable later.

## 4. The problem with the research question

Berger described this in **1929**. A paper reporting high accuracy on this
contrast contributes nothing.

> Speaker notes: Be direct here — this slide earns the audience's trust. State
> that if you had stopped at "I got 89% accuracy classifying brain states", the
> honest response would be "so did Hans Berger, without a computer." Then pivot:
> the undetermined question is *cost*, and cost is measurable.

## 5. Dataset and design

109 adults, 64 channels, 160 Hz, PhysioNet, open licence. Two 61-second baseline
runs each. Fully paired within-participant. 106 usable.

> Speaker notes: Emphasise the paired design — every participant is their own
> control, which removes the between-person variance that dominates EEG. Flag
> the confound honestly now rather than in the limitations: condition is
> confounded with run, because the two conditions are separate consecutive
> recordings. Saying this early makes the rest more credible.

## 6. Methods, and the one that matters

674 spectral features per 2-second epoch. Every split over **participants**,
never epochs.

> Speaker notes: Spend most of this slide on leakage. Each person contributes
> ~50 epochs sharing a skull and an electrode placement; a model that sees the
> same person on both sides identifies the *person*, not the brain state. Give
> the number: splitting epochs at random inflates AUC from 0.894 to 0.940. That
> single error produces more apparent improvement than the entire ML pipeline
> does honestly. This usually gets the biggest reaction of the talk.

## 7. The effect is real and it is huge

Occipital relative alpha increased on eye closure in **106 of 106**
participants. dz = +1.59. Occipital alpha power rises about fivefold.

> Speaker notes: Show the paired-lines panel. 106 out of 106 with no exceptions
> is rhetorically strong — let it land for a beat. Then set up the turn: "so the
> classification problem this defines is easy. Watch what happens when we ask
> what it costs."

## 8. Machine learning results — the deflation

| Feature set | ROC-AUC |
|---|---|
| 1 occipital alpha feature | 0.829 |
| 674 features, tuned model | 0.894 |

Nonlinear models: no advantage. 64 → 19 electrodes: costs 0.004.

> Speaker notes: This is the payload. Say the ratio out loud — two orders of
> magnitude more hardware and computation for about seven points of AUC. Then
> the two levers that *do* work: placement is worth +0.084 AUC at one electrode
> and nothing at nine, and one electrode with an 8-second window (0.856) matches
> 64 electrodes with a half-second window (0.857). Sixty-three electrodes and
> 7.5 seconds of latency are interchangeable.

## 9. Neuroscience interpretation, and what it is not

Posterior topography fits thalamocortical alpha generators. But: removing
**every** alpha feature costs 0.004 AUC.

> Speaker notes: The redundancy result is the subtle one. Alpha is *sufficient
> but not necessary* — eye closure reshapes the whole spectrum, so the
> information is duplicated across bands. This also explains why permutation
> importance ranks beta above alpha: on correlated features it measures
> redundancy, not relevance. Say explicitly that you are not claiming causation,
> and that the eyes-open/closed contrast differs in arousal as well as vision.

## 10. Conclusion and future work

Constrained by **physiology and recording time**, not by feature engineering or
model capacity. Spend the budget on placement and window length.

> Speaker notes: End on the honest caveat, because it is also the best future
> direction: these curves describe one large, focal, low-dimensional effect. For
> a subtler contrast — workload, attention, early pathology — complexity might
> well earn its keep, and nobody has measured that. Close with the framing from
> the paper: this study confirms a finding from 1929, and the confirmation is
> not the contribution. The measurement of what the modern apparatus adds is.

---

## Anticipated questions

**"Isn't 0.894 vs 0.829 a real improvement?"** Yes, and it is reported as one.
The claim is about the exchange rate, not the sign. Sixty-five thousandths of
AUC for 673 extra features and a tuned model is a poor trade if hardware cost
matters, and an excellent one if it does not.

**"Why not deep learning?"** With 5,698 epochs from 106 participants and a
linear model already at 0.894 against a 0.5 ceiling-adjusted headroom of 0.106,
there is very little left to capture. The two nonlinear models tested both
underperformed logistic regression.

**"Would a real single-electrode device do this well?"** Probably not. The
subsets are simulated by discarding channels from a 64-channel
average-referenced recording. A real device has a different reference,
different noise, and no neighbours to interpolate from. This is stated as a
limitation, not glossed.
