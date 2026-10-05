# GutSignal Lab

A technical and product prototype exploring one part of Suna Health's
signal-to-insight problem.

The project asks a narrow question:

> Can a small, reproducible baseline detect bowel-sound events in public
> abdominal audio and show when the result is uncertain?

This is an application portfolio project, not a medical device, diagnostic
system, or reconstruction of Suna's proprietary technology.

**Live demo:** https://gutsignal-lab.streamlit.app

## Deliverables

- A reproducible public-data audit and participant-aware evaluation
- An interpretable signal-processing and ML baseline
- An interactive explanation layer using non-diagnostic language
- An anonymised research synthesis about trust, usefulness, and uncertainty
- A concise model card and technical report

## Scientific principles

1. Split by participant wherever participant identifiers permit it.
2. Establish a simple baseline before attempting a complex model.
3. Report precision, recall, F1, average precision, and calibration, not only
   accuracy.
4. Inspect errors and possible leakage.
5. State clearly what the data and model cannot establish.

## Status

The baseline, interactive prototype, exploratory research, and public deployment
are complete.

## Engineering extension: robustness workbench

The **Robustness lab** lets a reviewer introduce controlled noise, gain changes,
clipping, missing signal, silence and constant DC into the three generated audio
examples. It compares scores and withholds inference when explicit quality checks
fail. It is a failure investigation, not a claim of device or clinical robustness.

The reproducible matrix covers all 33 example/condition combinations. Quality
checks withhold 15; five accepted cases change their displayed evidence state.
The largest accepted score shift is 95.4 percentage points under added noise.
This exposes a remaining weakness: ordinary-looking signal quality does not
guarantee a reliable model output. Gain changes of -12 and -30 dB leave these
example scores effectively unchanged, as expected from peak normalisation.

Uploads are decoded in memory with format, size, mono-channel, sample-rate,
duration and finite-value checks. The app does not write, log or cache uploaded
audio. Processing still takes place on the hosted server; this is not local-only
processing or a guarantee about the hosting provider's retention practices.
Use generated examples, not personal health recordings.

See [the engineering note](docs/ENGINEERING_NOTE.md) for thresholds, findings,
reproduction steps and limitations. The original model and grouped metrics have
not been replaced or improved by these demonstration experiments.

## Current result

The headline five-fold participant-grouped evaluation produces 0.945 average
precision, 0.825 ROC-AUC, and 0.896 F1. The dataset is event-heavy: always
predicting an event would produce 0.888 F1, so F1 alone overstates the useful
result. Balanced accuracy is 0.683 versus 0.500 for that constant prediction.
The Evidence tab makes this comparison visible. Random-recording validation is stronger,
so the project treats the grouped result as the more defensible estimate and
makes the generalisation gap visible.

The interface also withholds a confident result for scores between 35% and 65%.
In grouped cross-validation, this marks 217 of 1,606 recordings as uncertain and
captures 94 of the 276 forced-decision errors. Overall accuracy on the remaining
recordings rises from 82.8% to 86.9%. This internal result supports further
testing of abstention; it is not evidence of clinical safety.

## What the prototype does

The app takes a short WAV file, prepares the audio, extracts a compact set of
energy and frequency features, and passes those features to a logistic-regression
baseline. It then shows the resulting model-evidence score alongside the waveform,
spectrogram, and an uncertainty-aware explanation.

The public demonstration uses procedurally generated, non-human audio. Those
synthetic clips demonstrate how the interface behaves; their individual scores
are not evaluation evidence. The reported metrics come from cross-validation on
the attributed public research dataset.

## Three-minute reviewer path

1. Open **Try the prototype** and compare the stronger transient, quieter and
   ambiguous synthetic patterns.
2. Open **Evidence** to see why participant-grouped validation is treated as
   the headline result.
3. Open **Robustness lab**. Compare the original with noise, then silence or
   clipping. Inspect why a result is withheld and which failures remain.
4. Open **How it works** for the four-stage signal-to-insight pipeline.
5. Finish with **Limits & next steps** for the claims the prototype refuses to
   make and the findings from four exploratory participants.

The interface is intentionally demo-first. It should be understandable without
reading the code or assuming that a research probability is a health score.

## Reproduce the project

```bash
uv sync --extra dev
uv run python scripts/download_data.py
PYTHONPATH=src uv run python scripts/audit_dataset.py /path/to/data
PYTHONPATH=src uv run python scripts/build_features.py /path/to/data
PYTHONPATH=src uv run python scripts/train_baseline.py artifacts/features.csv
uv run python scripts/error_analysis.py
uv run python scripts/prepare_demo_samples.py
uv run python scripts/evaluate_robustness.py
uv run streamlit run streamlit_app.py
```

Run the checks with:

```bash
uv run ruff check .
uv run pytest -q
```

## Public deployment

The reviewer experience is live at https://gutsignal-lab.streamlit.app. It uses
`streamlit_app.py` as the entrypoint and Python 3.12. No API keys or private data
are required. The bundled model, evidence figure and three synthetic, non-human
demo clips are sufficient to run the reviewer experience.

## Project map

- `streamlit_app.py`: interactive evidence and explanation interface
- `src/gutsignal/`: dataset, feature and modelling code
- `scripts/`: reproducible data, training and analysis commands
- `docs/MODEL_CARD.md`: intended use, evaluation and limitations
- `docs/TECHNICAL_REPORT.md`: technical and product narrative
- `docs/ENGINEERING_NOTE.md`: input quality, perturbations and failure investigation
- `docs/INTERVIEW_GUIDE.md`: lightweight potential-user research
- `docs/DATA_AND_ATTRIBUTION.md`: provenance, licence and grouping caveat
- `docs/PUBLISHING_CHECKLIST.md`: repository and deployment handoff

Internal application materials, raw responses, the fuller research memo and the
optional walkthrough script are excluded from the public repository.

## Boundaries

GutSignal Lab is an independent, non-commercial portfolio experiment. It is
not affiliated with Suna Health, is not a medical device and does not evaluate
digestive health. See the model card and data-attribution note before reusing
the model or included audio.

## Licence

The original source code is available under the MIT License. The bundled demo
audio is generated specifically for this repository and contains no participant
recordings. The trained model is retained for this non-commercial portfolio
demonstration; review the source dataset terms before any reuse. See `LICENSE` and
`docs/DATA_AND_ATTRIBUTION.md` for the exact boundary.
