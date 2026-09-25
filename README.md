# GutSignal Lab

An independent technical and product prototype exploring Suna Health's
signal-to-insight problem.

The project asks a deliberately narrow question:

> Can a small, reproducible baseline detect bowel-sound events in public
> abdominal audio while communicating uncertainty honestly?

This is an application portfolio project, not a medical device, diagnostic
system, or reconstruction of Suna's proprietary technology.

**Live demo:** https://gutsignal-lab.streamlit.app

## Deliverables

- A reproducible public-data audit and participant-aware evaluation
- A transparent signal-processing and ML baseline
- An interactive explanation layer using non-diagnostic language
- A short user-research memo about trust, usefulness, and uncertainty
- A concise model card and technical report

## Scientific principles

1. Split by participant wherever participant identifiers permit it.
2. Establish a simple baseline before attempting a complex model.
3. Report precision, recall, F1, average precision, and calibration, not only
   accuracy.
4. Inspect errors and possible leakage.
5. State clearly what the data and model cannot establish.

## Status

Working baseline and reviewer-facing interactive prototype complete and
publicly deployed. User research and final application packaging are in progress.

## Current result

The headline five-fold participant-grouped evaluation produces 0.945 average
precision, 0.825 ROC-AUC, and 0.896 F1. Random-recording validation is stronger,
so the project treats the grouped result as the more defensible estimate and
makes the generalisation gap visible.

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
   deliberately ambiguous synthetic patterns.
2. Open **Evidence** to see why participant-grouped validation is treated as
   the headline result.
3. Open **How it works** for the four-stage signal-to-insight pipeline.
4. Finish with **Limits & next steps** for the claims the prototype refuses to
   make and the user-research questions still being tested.

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
- `docs/INTERVIEW_GUIDE.md`: lightweight potential-user research
- `docs/DATA_AND_ATTRIBUTION.md`: provenance, licence and grouping caveat
- `docs/PUBLISHING_CHECKLIST.md`: repository and deployment handoff

Internal application materials, interview responses and the walkthrough script
are intentionally excluded from the public repository.

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
