from __future__ import annotations

import json
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gutsignal.audio import AudioInputError, decode_wav, encode_wav
from gutsignal.features import preprocess_signal
from gutsignal.inference import evidence_state, predict_clip
from gutsignal.modeling import class_balance_reference
from gutsignal.robustness import perturb

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "artifacts" / "model" / "baseline.joblib"
METRICS_PATH = ROOT / "artifacts" / "model" / "metrics.json"
EVIDENCE_IMAGE_PATH = ROOT / "artifacts" / "report" / "model_evidence.png"
DEMO_SAMPLES_PATH = ROOT / "demo_samples"
ROBUSTNESS_PATH = ROOT / "artifacts" / "report" / "robustness.json"


st.set_page_config(page_title="GutSignal Lab", page_icon="〰️", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #17342c;
        --muted: #60716a;
        --surface: #f5f8f3;
        --line: #dce6df;
        --accent: #ff7e68;
    }
    .stApp { background: #fbfcf9; }
    .block-container { max-width: 1180px; padding-top: 2.2rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
    .hero-copy {
        color: #40564e;
        font-size: 1.15rem;
        line-height: 1.65;
        max-width: 700px;
    }
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.7rem;
        margin-top: 0.6rem;
    }
    .stat-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem;
    }
    .stat-value { color: var(--ink); font-size: 1.35rem; font-weight: 750; }
    .stat-label { color: var(--muted); font-size: 0.78rem; margin-top: 0.2rem; }
    .section-note {
        background: var(--surface);
        border-left: 4px solid var(--accent);
        border-radius: 6px;
        color: #40564e;
        margin: 0.8rem 0 1.2rem;
        padding: 0.9rem 1rem;
    }
    .footer-note {
        border-top: 1px solid var(--line);
        color: var(--muted);
        font-size: 0.82rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
    }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 0.9rem;
    }
    @media (max-width: 700px) {
        .stat-grid { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_bundle() -> dict:
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_demo_manifest() -> dict:
    return json.loads((DEMO_SAMPLES_PATH / "manifest.json").read_text(encoding="utf-8"))


def waveform_chart(signal: np.ndarray, sample_rate: int) -> go.Figure:
    time = np.arange(len(signal)) / sample_rate
    figure = go.Figure(go.Scatter(x=time, y=signal, mode="lines", line={"color": "#ff7e68"}))
    figure.update_layout(
        height=260,
        margin={"l": 30, "r": 20, "t": 20, "b": 30},
        xaxis_title="Time (seconds)",
        yaxis_title="Normalised amplitude",
        template="plotly_white",
    )
    return figure


def spectrogram_chart(signal: np.ndarray, sample_rate: int) -> go.Figure:
    spectrum = np.abs(librosa.stft(signal, n_fft=512, hop_length=128))
    db = librosa.amplitude_to_db(spectrum, ref=np.max)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=512)
    times = librosa.frames_to_time(np.arange(db.shape[1]), sr=sample_rate, hop_length=128)
    keep = frequencies <= 2_000
    figure = go.Figure(
        go.Heatmap(
            x=times,
            y=frequencies[keep],
            z=db[keep],
            colorscale="Magma",
            colorbar={"title": "dB"},
        )
    )
    figure.update_layout(
        height=320,
        margin={"l": 30, "r": 20, "t": 20, "b": 30},
        xaxis_title="Time (seconds)",
        yaxis_title="Frequency (Hz)",
        template="plotly_white",
    )
    return figure


def interpretation(probability: float) -> tuple[str, str]:
    state = evidence_state(probability)
    explanations = {
        "Lower model evidence": "The baseline found less evidence of an annotated bowel-sound event in this clip.",
        "Uncertain": "The result sits near the decision boundary and should not be treated as a confident detection.",
        "Higher model evidence": "The baseline found more evidence of a sound pattern resembling expert-annotated events in the public dataset.",
    }
    return state, explanations[state]


hero, snapshot = st.columns([1.45, 1], gap="large")
with hero:
    st.title("GutSignal Lab")
    st.markdown(
        '<div class="hero-copy">Can a simple model identify bowel-sound events in public '
        "abdominal audio and explain when it is uncertain?</div>",
        unsafe_allow_html=True,
    )
with snapshot:
    st.markdown(
        """
        <div class="stat-grid">
            <div class="stat-card"><div class="stat-value">1,606</div><div class="stat-label">labelled clips</div></div>
            <div class="stat-card"><div class="stat-value">19</div><div class="stat-label">reported participants</div></div>
            <div class="stat-card"><div class="stat-value">5-fold</div><div class="stat-label">grouped validation</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.info(
    "Research demonstration only. This prototype detects whether a two-second audio clip resembles "
    "annotated bowel-sound events in one small public dataset. It does not assess digestive health, "
    "diagnose a condition, or reproduce Suna Health's technology."
)

demo_tab, robustness_tab, evidence_tab, method_tab, limits_tab = st.tabs(
    ["Try the prototype", "Robustness lab", "Evidence", "How it works", "Limits & next steps"]
)

with demo_tab:
    st.subheader("See how the model responds to a short recording")
    st.markdown(
        '<div class="section-note"><strong>What to do:</strong> choose one of the generated clips, '
        "listen to it, and compare the score with the waveform and spectrogram. These examples show "
        "how the app works. Their scores do not measure the model's accuracy.</div>",
        unsafe_allow_html=True,
    )
    mode = st.radio(
        "Choose an input", ["Curated synthetic example", "Upload a WAV"], horizontal=True
    )
    payload: bytes | None = None
    example_label: str | None = None

    if mode == "Curated synthetic example":
        manifest = load_demo_manifest()
        example_name = st.selectbox("Example", list(manifest))
        example = manifest[example_name]
        selected_path = DEMO_SAMPLES_PATH / example["file"]
        payload = selected_path.read_bytes()
        example_label = example["example_label"]
        st.caption(example["purpose"])
        st.audio(payload, format="audio/wav")
    else:
        uploaded = st.file_uploader("Upload a two-second WAV recording", type=["wav"])
        st.caption(
            "Two-second mono WAV, maximum 8 MiB. Processing runs on the hosted server, "
            "not on your device. This app does not save, log or cache uploaded audio; "
            "Streamlit holds it in session memory. Do not upload personal health recordings."
        )
        if uploaded is not None:
            payload = uploaded.getvalue()

    if payload is not None:
        try:
            clip = decode_wav(payload)
            result = predict_clip(clip, load_model_bundle())
            if result.probability is None:
                st.warning("Result withheld: input quality. " + " ".join(result.quality.reasons))
            else:
                signal, sample_rate = preprocess_signal(clip.signal, clip.sample_rate)
                label, explanation = interpretation(result.probability)
                left, right = st.columns([1, 2])
                left.metric("Model evidence", f"{result.probability:.1%}")
                right.markdown(f"### {label}\n{explanation}")
                if example_label is not None:
                    st.caption(f"Synthetic pattern description: **{example_label}**")
                st.plotly_chart(waveform_chart(signal, sample_rate), width="stretch")
                st.plotly_chart(spectrogram_chart(signal, sample_rate), width="stretch")
                st.caption(
                    "The probability is a research-model output, not a health score. Passing the "
                    "input checks does not establish that this is abdominal audio or that it "
                    "resembles the training data."
                )
        except AudioInputError as error:
            st.warning(str(error))
        except (ValueError, RuntimeError):
            st.error("No result: audio processing or model compatibility check failed.")

with robustness_tab:
    st.subheader("What happens when the recording changes?")
    st.write(
        "A confident score can still be wrong. Change a generated recording and compare the "
        "model's response. Basic quality checks withhold a result for silence, strong clipping "
        "or flat segments. They do not detect every kind of noise or movement."
    )
    st.caption(
        "Generated, non-human audio only. This is a deterministic engineering experiment, "
        "not new participant validation, a device simulation or an accuracy benchmark."
    )
    controls, comparison = st.columns([1, 2], gap="large")
    with controls:
        lab_name = st.selectbox("Workbench example", list(load_demo_manifest()))
        kind = st.selectbox(
            "Introduce a disturbance",
            ["Noise", "Gain", "Clipping", "Dropout", "Silence", "Constant DC", "Unchanged"],
        )
        if kind == "Noise":
            level = st.slider("Signal-to-noise ratio (dB)", 0, 40, 10, 5)
            st.caption("Lower means stronger noise relative to the signal.")
        elif kind == "Gain":
            level = st.slider("Gain change (dB)", -60, 0, -12, 3)
        elif kind == "Clipping":
            level = st.slider("Gain before clipping", 1, 20, 8)
        elif kind == "Dropout":
            level = st.slider("Missing signal (%)", 0, 100, 25, 5)
        else:
            level = 0
    lab_file = DEMO_SAMPLES_PATH / load_demo_manifest()[lab_name]["file"]
    clean_clip = decode_wav(lab_file.read_bytes())
    changed_clip = perturb(clean_clip, kind, float(level))
    clean_result = predict_clip(clean_clip, load_model_bundle())
    changed_result = predict_clip(changed_clip, load_model_bundle())
    with comparison:
        original, changed = st.columns(2)
        original.metric("Original score", f"{clean_result.probability:.1%}")
        original.caption(clean_result.state)
        if changed_result.probability is None:
            changed.metric("Changed score", "Withheld")
            st.warning(" ".join(changed_result.quality.reasons))
        else:
            shift = (changed_result.probability - clean_result.probability) * 100
            changed.metric(
                "Changed score",
                f"{changed_result.probability:.1%}",
                delta=f"{shift:+.1f} percentage points",
                delta_color="off",
            )
            changed.caption(changed_result.state)
            if changed_result.state != clean_result.state:
                st.warning("The displayed evidence state changed under this disturbance.")
        st.audio(encode_wav(changed_clip), format="audio/wav")
        time = np.arange(clean_clip.signal.size) / clean_clip.sample_rate
        overlay = go.Figure()
        overlay.add_scatter(x=time, y=clean_clip.signal, name="Original", line={"color": "#17342c"})
        overlay.add_scatter(
            x=time,
            y=changed_clip.signal,
            name="Changed",
            opacity=0.65,
            line={"color": "#ff7e68"},
        )
        overlay.update_layout(
            height=260,
            template="plotly_white",
            xaxis_title="Time (seconds)",
            yaxis_title="Raw amplitude",
            margin={"l": 20, "r": 20, "t": 20, "b": 30},
        )
        st.plotly_chart(overlay, width="stretch")
    with st.expander("Inspect the input checks and their limits"):
        st.write(
            "Checks run before peak normalisation: duration 1.9 to 2.1 seconds; centred RMS "
            "at least 0.00001; no more than 1% of samples at magnitude 0.999 or above; "
            "fewer than 20% flat 20 ms windows. These manually chosen thresholds are "
            "engineering heuristics, not calibrated quality or out-of-distribution detectors."
        )
        st.write(
            "Peak normalisation should make gain changes mostly disappear. Added noise changes "
            "spectral features and can move the score even when every quality check passes. "
            "That remaining failure mode is deliberately visible here."
        )
    st.subheader("The reproducible stress matrix")
    report = json.loads(ROBUSTNESS_PATH.read_text())
    summary = report["summary"]
    matrix_columns = st.columns(3)
    matrix_columns[0].metric("Generated test cases", str(summary["cases"]))
    matrix_columns[1].metric("Withheld by quality checks", str(summary["withheld"]))
    matrix_columns[2].metric("Accepted state changes", str(summary["accepted_state_changes"]))
    matrix = pd.DataFrame(report["cases"])
    visible = matrix[["example", "condition", "score", "change_percentage_points", "state"]]
    st.dataframe(
        visible.style.format(
            {"score": "{:.1%}", "change_percentage_points": "{:+.1f}"}, na_rep="Withheld"
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "All 11 conditions are reported for all three examples, including failures. State changes "
        "are sensitivity observations, not labelled errors. Original grouped metrics are unchanged."
    )
    st.download_button(
        "Download stress matrix (JSON)",
        ROBUSTNESS_PATH.read_bytes(),
        file_name="GutSignal_Robustness.json",
        mime="application/json",
    )

with evidence_tab:
    metrics = load_metrics()
    grouped = metrics["participant_grouped_cross_validation"]
    random = metrics["random_recording_cross_validation"]
    uncertainty = metrics["uncertainty_band_evaluation"]

    st.subheader("Participant-grouped validation")
    st.write(
        "Recordings from the same inferred participant group stay together during validation. "
        "This is harder than randomly mixing clips and gives a more realistic test of whether "
        "the model might work for people it has not seen before."
    )

    columns = st.columns(4)
    columns[0].metric(
        "Average precision",
        f"{grouped['average_precision']:.3f}",
        help="Summarises precision and recall across thresholds; higher is better.",
    )
    columns[1].metric(
        "ROC-AUC",
        f"{grouped['roc_auc']:.3f}",
        help="Measures ranking performance across classification thresholds.",
    )
    columns[2].metric(
        "F1",
        f"{grouped['f1']:.3f}",
        help="Harmonic mean of precision and recall at the selected threshold.",
    )
    columns[3].metric(
        "Brier score",
        f"{grouped['brier_score']:.3f}",
        help="Measures probability error; lower is better.",
    )

    comparison = pd.DataFrame(
        {
            "Evaluation": ["Participant-grouped", "Random recording"],
            "Average precision": [grouped["average_precision"], random["average_precision"]],
            "ROC-AUC": [grouped["roc_auc"], random["roc_auc"]],
            "F1": [grouped["f1"], random["f1"]],
        }
    )
    st.dataframe(
        comparison.style.format(
            {"Average precision": "{:.3f}", "ROC-AUC": "{:.3f}", "F1": "{:.3f}"}
        ),
        hide_index=True,
        width="stretch",
    )
    st.warning(
        "Randomly mixing recordings produced higher scores, but that easier test can overestimate "
        "real-world performance. The participant-grouped result above is the more cautious estimate."
    )

    st.subheader("Why F1 needs a reference")
    counts = metrics["dataset"]
    reference = class_balance_reference(counts["event_positive"], counts["event_negative"])
    reference_frame = pd.DataFrame(
        {
            "Method": ["Always predict event", "Participant-grouped model"],
            "Accuracy": [reference["accuracy"], grouped["accuracy"]],
            "Balanced accuracy": [reference["balanced_accuracy"], grouped["balanced_accuracy"]],
            "F1": [reference["f1"], grouped["f1"]],
        }
    )
    st.dataframe(
        reference_frame.style.format(
            {"Accuracy": "{:.3f}", "Balanced accuracy": "{:.3f}", "F1": "{:.3f}"}
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"{counts['event_positive']:,} of {counts['recordings']:,} clips contain events. "
        f"Always predicting event gives F1 {reference['f1']:.3f}, close to the model's "
        f"{grouped['f1']:.3f}, but detects no non-event clips. The grouped model's balanced "
        "accuracy is more informative here. This reference is computed from the reported "
        "class counts, not from a new training or evaluation run."
    )

    st.subheader('When the model is allowed to say "uncertain"')
    st.write(
        "The interface withholds a confident result for scores between 35% and 65%. "
        "Participant-grouped evaluation shows whether that range actually concentrates mistakes."
    )
    uncertainty_columns = st.columns(3)
    uncertainty_columns[0].metric(
        "Marked uncertain",
        f"{uncertainty['uncertain_fraction']:.1%}",
        help=(
            f"{uncertainty['uncertain_recordings']:,} of {uncertainty['recordings']:,} recordings"
        ),
    )
    uncertainty_columns[1].metric(
        "Accuracy on remaining recordings",
        f"{uncertainty['decided_accuracy']:.1%}",
        delta=(
            f"{(uncertainty['decided_accuracy'] - grouped['accuracy']) * 100:.1f} percentage points"
        ),
        help="Overall accuracy after the uncertain recordings are withheld.",
    )
    uncertainty_columns[2].metric(
        "Errors inside uncertain range",
        f"{uncertainty['error_capture_rate']:.1%}",
        help=(
            f"{uncertainty['errors_inside_band']} of the forced decisions that would have been wrong"
        ),
    )
    st.caption(
        "This is an internal cross-validation result, not proof of clinical safety. Overall accuracy "
        "improves, but balanced accuracy changes little because the dataset contains many more event "
        "clips than non-event clips. The band concentrates some errors and still requires external "
        "validation."
    )
    st.image(EVIDENCE_IMAGE_PATH, width="stretch")

with method_tab:
    st.subheader("A simple, auditable baseline")
    st.write(
        "The goal was to build a reproducible first system whose assumptions, failure modes and "
        "next experiments could be inspected."
    )

    step_columns = st.columns(4)
    steps = [
        (
            "01 · Prepare",
            "Resample to 8 kHz, remove DC offset, band-pass 40–2,000 Hz and peak-normalise.",
        ),
        (
            "02 · Describe",
            "Extract energy, zero-crossing, spectral and MFCC summary features.",
        ),
        (
            "03 · Model",
            "Fit a regularised logistic-regression baseline with probability outputs.",
        ),
        (
            "04 · Challenge",
            "Compare five-fold participant-grouped validation with a random recording split.",
        ),
    ]
    for column, (title, copy) in zip(step_columns, steps, strict=True):
        with column.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(copy)

    st.subheader("Why start simple?")
    st.markdown(
        """
        - With a small public dataset, added model complexity would not remove the generalisation problem.
        - Interpretable features make errors and shortcuts easier to investigate.
        - This baseline provides a reference point for any later deep-learning model.
        - Product language and abstention behaviour matter alongside discrimination metrics.
        """
    )

with limits_tab:
    st.subheader("A detection is not a diagnosis")
    st.markdown(
        """
        This baseline cannot establish:

        - whether someone's digestion is healthy or unhealthy;
        - whether a meal caused a symptom;
        - whether a person has any gastrointestinal condition;
        - whether performance transfers to a different contact microphone or wearable;
        - whether the model remains reliable during movement or ordinary daily life.

        The public dataset contains 19 reported participants and was recorded with specialised equipment.
        A real product would require broader collection, prospective validation, robust artefact testing,
        carefully defined ground truth, and clinical and regulatory review of every user-facing claim.
        """
    )
    st.subheader("What I would test next")
    st.markdown(
        """
        1. Validate on entirely new participants and recording sessions.
        2. Extend the generated-audio stress tests to labelled participant recordings, real motion,
           speech, clothing friction and imperfect sensor contact.
        3. Compare performance across devices and skin-placement variation.
        4. Confirm the uncertainty range on an external participant holdout and define when the model
           must withhold a result.
        5. Co-design explanations with users and clinicians before exposing outputs.
        """
    )

    st.subheader("What four exploratory conversations revealed")
    st.write(
        "Four adults answered the same five anonymous questions. This was a small convenience "
        "sample for product discovery, not representative user validation. No diagnoses, medical "
        "records, names, or contact details were collected, and no direct quotations are published."
    )
    research_columns = st.columns(3)
    research_columns[0].markdown(
        "**Spot patterns**\n\nAll four described looking backwards at meals or timing. The clearest "
        "job is helping people notice patterns and possible food relationships, not issuing a "
        "single health score."
    )
    research_columns[1].markdown(
        "**Earn trust**\n\nParticipants wanted evidence, explanations of how information is "
        "collected, and a simple interface. Flashy claims or unrelated data use would reduce trust."
    )
    research_columns[2].markdown(
        "**Explain uncertainty**\n\nPeople wanted a reason for uncertainty and a sensible next step. "
        "Clinical signposting should be proportionate, not attached to every uncertain result."
    )
    st.info(
        "The research also showed a gap between what people wanted and what this model can support. "
        "Participants asked for nutrient deficiencies, digestion speed, food triggers, and advice "
        "on what to avoid. This prototype cannot support those conclusions, so they remain out of scope."
    )

    st.subheader("Read the work")
    left, middle, right = st.columns(3)
    left.download_button(
        "Download the model card",
        data=(ROOT / "docs" / "MODEL_CARD.md").read_text(),
        file_name="GutSignal_Model_Card.md",
        mime="text/markdown",
    )
    middle.download_button(
        "Download the technical report",
        data=(ROOT / "docs" / "TECHNICAL_REPORT.md").read_text(),
        file_name="GutSignal_Technical_Report.md",
        mime="text/markdown",
    )
    right.download_button(
        "Review data attribution",
        data=(ROOT / "docs" / "DATA_AND_ATTRIBUTION.md").read_text(),
        file_name="GutSignal_Data_Attribution.md",
        mime="text/markdown",
    )

st.markdown(
    '<div class="footer-note">Portfolio project. '
    "Public data and generated examples · non-diagnostic · October 2026</div>",
    unsafe_allow_html=True,
)
