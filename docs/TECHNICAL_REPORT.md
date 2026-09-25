# GutSignal Lab: from physiological sound to honest product insight

## Executive summary

GutSignal Lab is an independent, time-boxed application sprint inspired by
Suna Health's signal-to-insight problem. The sprint does not attempt to recreate
Suna's product. It asks a narrower question: can a small, reproducible baseline
detect expert-annotated bowel-sound events in public abdominal audio, and can
the result be communicated without turning a model output into a health claim?

The project covers public-data auditing, audio preprocessing, feature
engineering, participant-aware validation, error analysis, an interactive
interface, and lightweight user research. The central finding is methodological:
validation design changes the apparent result. Randomly allocating recordings
across folds produces better scores than keeping participant groups separate.
The latter is treated as the headline result.

## Data and target

The public dataset contains 1,606 two-second, 44.1 kHz contact-microphone
recordings with corresponding CSV annotations. A recording is labelled positive
when its annotation file contains at least one expert-marked bowel-sound event.
There are 1,283 positive and 323 negative recordings.

The source paper reports 19 participants. Filename suffixes produce exactly 19
groups, so the suffix is used as a provisional participant identifier. This is
explicitly documented as an inference pending author confirmation.

## Baseline method

Recordings are resampled to 8 kHz, centred, band-pass filtered between 40 and
2,000 Hz, and peak normalised. Features summarise energy, zero crossings,
spectral shape, low/high-frequency energy, and MFCCs. A regularised logistic
regression provides an intentionally interpretable first baseline.

This model is not intended to maximise a leaderboard metric. Its purpose is to
establish a transparent reference point, expose validation risks, and support
clear error analysis before trying more complex architectures.

## Results

The headline five-fold stratified group evaluation keeps inferred participants
separate across training and evaluation folds.

| Metric | Grouped | Random recording |
|---|---:|---:|
| Accuracy | 0.828 | 0.860 |
| Balanced accuracy | 0.683 | 0.732 |
| F1 | 0.896 | 0.915 |
| Average precision | 0.945 | 0.970 |
| ROC-AUC | 0.825 | 0.896 |
| Brier score | 0.123 | 0.099 |

The grouped confusion matrix contains 1,188 true positives, 142 true negatives,
181 false positives, and 95 false negatives at a fixed threshold of 0.5. The
result is promising for a simple baseline, but the false-positive count and
calibration curve argue against interpreting the output as a reliable health
signal.

### Initial abstention analysis

The interface marks probabilities from 0.35 up to, but not including, 0.65 as
uncertain. On participant-grouped out-of-fold predictions, that range contains
217 of 1,606 recordings and 94 of the 276 forced-decision errors. Overall accuracy
on the remaining recordings rises from 82.8% to 86.9%.

This is evidence that the range concentrates some mistakes, but the result is
not conclusive. Balanced accuracy changes only from 0.683 to 0.685, partly because
the dataset contains many more event clips than non-event clips. External
participant, device, and recording-condition validation remains necessary.

## Product interpretation

The interface uses three states: lower evidence, uncertain, and higher evidence.
It avoids phrases such as "healthy gut" or "abnormal digestion." The model only
answers whether a short clip resembles event annotations in the training data.
It cannot establish causality, digestive quality, food response, or disease.

This distinction is central to the prototype. In health technology, a polished
interface can create more confidence than the evidence warrants. The design
therefore makes limitations visible at the moment a result is shown.

## What I would test next

1. Confirm the participant identifier and use a locked, external participant
   holdout.
2. Collect broader data across devices, placements, body positions, and daily
   environments.
3. Create explicit artefact sets for motion, clothing, speech, contact loss,
   and ambient sound.
4. Compare the transparent baseline with compact CNN and self-supervised audio
   representations.
5. Confirm the uncertainty range on an external participant holdout and define
   when the model must withhold a result.
6. Co-design language with users and clinicians, then measure comprehension
   rather than assuming that explanations work.
7. Define product claims only after linking each claim to suitable ground truth
   and prospective validation.

## Why this sprint matters

The technical model is only one part of the work. The stronger demonstration is
the operating approach: reduce an ambiguous problem to a defensible question,
build the simplest useful system, find where the attractive result is fragile,
communicate that honestly, and identify the next experiment.
