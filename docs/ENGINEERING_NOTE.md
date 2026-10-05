# Engineering extension: challenge the input before trusting the score

## The problem

The original prototype passed a WAV directly through feature extraction and the
model. Invalid or very short audio could fail during preprocessing. Silence or
distorted audio could still produce a numerical score. Uploaded files were
temporarily written to disk and cleanup happened only after successful inference.

The extension makes input handling explicit and adds an interactive workbench
to investigate whether scores change under controlled disturbances. This is a
small engineering increment, not production certification or deeper clinical ML.

## What changed

- In-memory WAV decoding, with limits checked before reading sample arrays.
- An 8 MiB payload limit and a matching Streamlit upload limit.
- A mono-only input contract, 8 to 192 kHz sample rates and 1.9 to 2.1 s duration.
- Rejection of malformed content, non-WAV content and non-finite numeric samples.
- Quality checks on raw samples, before peak normalisation can hide low level.
- A separate result-withheld state that does not call feature extraction or the model.
- Shared preprocessing and feature extraction for file-based training and in-memory inference.
- A feature-schema compatibility check before prediction.
- A deterministic workbench, a downloadable full stress matrix and model/input hashes.
- Regression tests including inference parity with the original path.
- Existing GitHub Actions strengthened with Python 3.12 and frozen dependencies.
- A count-derived always-event reference to explain the effect of class imbalance on F1.
- A regression check that regenerates all demo WAVs byte-for-byte and verifies their manifest.

Uploads are not persisted or cached by the application. They are processed on
the hosted server and remain in Streamlit session memory; no claim is made about
infrastructure retention. The UI asks reviewers not to upload personal health audio.
The workbench uses only the bundled, generated non-human recordings.

## Quality rules

| Check | Withhold when | Important limitation |
|---|---|---|
| Duration | Outside 1.9 to 2.1 seconds | Model target was defined on two-second clips |
| Centred RMS | Below 0.00001 | Uncalibrated digital level, not physical loudness |
| Full-scale samples | More than 1% have magnitude at least 0.999 | Possible clipping, not proof of sensor saturation |
| Flat segments | At least 20% of 20 ms windows have range below 0.00000001 | May reject genuine quiet intervals; short/noisy dropouts can pass |

These thresholds were chosen manually for prototype input handling. They have
not been fitted or validated against labelled participant-quality annotations.
Their false-rejection rate on the public dataset is unknown. Passing them does
not show that audio is abdominal, in distribution, useful, or clinically safe.

## Reproducible experiment

Each of the three generated examples is tested under all eleven conditions:
unchanged; -12 and -30 dB gain; additive white noise at 20, 10 and 0 dB SNR;
8x gain clipped to the digital range; 25% and 50% centred dropout; silence;
and constant DC. Seed 42 makes the noise reproducible. Added noise is scaled
with the signal if necessary to avoid digital clipping without changing SNR.

The floating-point perturbations do not simulate real sensor acquisition,
quantisation, physiology, contact microphones, speech, motion or clothing friction.
They isolate simple signal changes. All conditions, not only favourable ones,
appear in `artifacts/report/robustness.json`.

## Findings

- 33 example/condition combinations were run.
- 15 were withheld by quality checks: clipping, both dropout levels, silence
  and constant DC for each example.
- Five of the 18 accepted cases changed the displayed evidence state.
- On the event-like example, the score fell from 98.2% to 21.7% at 10 dB SNR,
  and to 2.8% at 0 dB SNR, despite passing the quality checks.
- The largest accepted absolute change was 95.4 percentage points.
- -12 and -30 dB gain alone left example scores effectively unchanged, as
  expected because preprocessing peak-normalises the input.

These are sensitivity findings, not accuracy or error-rate estimates. Generated
pattern names are not expert labels. The original grouped evaluation metrics and
trained model are unchanged. No extra participants or prospective validation are
implied. The unguarded path is reproduced only in parity tests, not exposed as a
health decision in the interface.

## What this leaves unresolved

The original model's F1 is 0.896, while always predicting an event would give
0.888 on the reported class counts. The grouped model's balanced accuracy of
0.683, versus 0.500 for that constant prediction, is a more useful part of the
result. The reference is computed from counts; it is not a new model experiment.
The UI now displays it alongside the grouped result to prevent overstating F1.

Noise can pass the quality gate and push the model toward a confident state.
The existing 35% to 65% evidence band will not catch every such case. A next
research iteration should measure quality gating and perturbation effects on
participant-held-out labelled audio, including false rejections and subgroup
variation. Thresholds or augmentations must be selected without using evaluation
fold labels; real device data would still be required for device-transfer claims.

## Reproduce and inspect

```bash
uv sync --frozen --extra dev
uv run python scripts/evaluate_robustness.py
uv run ruff check .
uv run pytest -q
uv run streamlit run streamlit_app.py
```

The report records model and input SHA-256 hashes. Tests recompute the matrix,
compare states and scores, and ensure quality-withheld inputs never reach the
model. The web tests exercise the workbench and its silence state.

Implementation references:

- [SoundFile in-memory audio and bounded SoundFile reads](https://python-soundfile.readthedocs.io/en/latest/)
- [Streamlit upload limits and in-memory UploadedFile interface](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)
