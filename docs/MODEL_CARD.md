# Model card: GutSignal event-detection baseline

## Summary

GutSignal is a deliberately small research baseline that estimates whether a
two-second abdominal audio clip contains a sound event resembling the expert
annotations in one public bowel-sound dataset.

It is a portfolio demonstration of reproducible modelling and responsible
communication. It is not a health score, diagnostic system, medical device, or
reconstruction of Suna Health's proprietary technology.

## Model details

- **Model:** Logistic regression
- **Input:** Two-second mono WAV recording
- **Preprocessing:** 8 kHz resampling, DC removal, sixth-order 40–2,000 Hz
  Butterworth band-pass filter, peak normalisation
- **Features:** RMS amplitude, zero-crossing rate, spectral centroid,
  bandwidth, roll-off, flatness, low/high-band energy fractions, and 13 MFCC
  means and standard deviations
- **Output:** Probability-like score for the presence of an expert-annotated
  bowel-sound event
- **Decision threshold:** 0.5 for the documented baseline

The probability should not be interpreted as clinical risk or digestive
health. Class weighting was intentionally not used in the headline model
because it worsened probability calibration in preliminary comparison.

## Dataset

The project uses the public **Bowel sounds** dataset by Robert Nowak, associated
with Ficek et al. (2021). It contains 1,606 two-second recordings and paired
expert annotation files. The research reports 19 participants.

The filename suffix produces exactly 19 groups. This sprint uses that suffix as
the participant grouping variable for cross-validation. That mapping is a
reasonable but documented inference and should be verified with the dataset
authors before publication-quality work.

The dataset is licensed CC BY-NC 4.0. This non-commercial job-application
project attributes the source. Commercial reuse of the data or derived model
would require a separate licence review.

## Headline evaluation

Five-fold stratified group cross-validation keeps inferred participant groups
out of one another's training and evaluation folds.

| Metric | Participant-grouped result |
|---|---:|
| Accuracy | 0.828 |
| Balanced accuracy | 0.683 |
| Precision | 0.868 |
| Recall | 0.926 |
| F1 | 0.896 |
| Average precision | 0.945 |
| ROC-AUC | 0.825 |
| Brier score | 0.123 |

Random-recording cross-validation produces higher results, including ROC-AUC
of 0.896. The grouped result is used as the headline estimate because random
clip allocation can expose a model to recordings from the same person during
training and evaluation.

## Uncertainty range

The interface labels probabilities from 0.35 up to, but not including, 0.65 as
uncertain. Applied to participant-grouped out-of-fold predictions, this range:

- withholds a confident result for 217 of 1,606 recordings (13.5%);
- contains 94 of the 276 forced-decision errors (34.1%);
- raises overall accuracy on the remaining recordings from 82.8% to 86.9%; and
- produces a 43.3% forced-decision error rate inside the uncertain range.

Balanced accuracy on the decided subset is 0.685, compared with 0.683 across all
recordings. The overall accuracy increase partly reflects class imbalance, so the
result should be interpreted as evidence that the band concentrates some errors,
not as proof of calibration, clinical safety, or reliable abstention on new data.

## Intended uses

- Demonstrating an end-to-end audio-ML workflow
- Exploring validation choices in small physiological datasets
- Prototyping uncertainty-aware product explanations
- Supporting discussion about the difference between event detection and
  health inference

## Out-of-scope uses

- Diagnosing, screening, treating, or monitoring a medical condition
- Assessing whether someone's digestion is healthy
- Attributing symptoms to food, sleep, stress, or another cause
- Comparing individuals
- Making decisions about diet, medication, or care
- Representing expected performance on Suna's hardware

## Known limitations

1. Only 19 reported participants are represented.
2. The recordings come from specialised equipment and controlled collection;
   generalisation to phones or wearables is unknown.
3. Performance may change with movement, clothing friction, speech, ambient
   noise, body position, placement, and sensor pressure.
4. The binary target detects annotated events; it does not measure digestive
   function or explain why an event occurred.
5. Labels depend on expert judgement and may contain uncertainty.
6. The threshold was not selected for a clinical operating point.
7. Calibration remains imperfect. The uncertainty region concentrates some
   internal cross-validation errors but has not been validated externally.

## Responsible presentation

The interface says "model evidence," not "gut health" or "diagnosis." It
contains an explicit uncertainty state and explains possible confounders. A
real consumer product would require prospective data collection, external
validation, artefact stress-testing, calibrated abstention, clinician review,
and careful governance of every claim.

## Source

Ficek J, Radzikowski K, Nowak JK, Yoshie O, Walkowiak J, Nowak R. Analysis of
Gastrointestinal Acoustic Activity Using Deep Neural Networks. *Sensors*.
2021;21(22):7602. https://doi.org/10.3390/s21227602
