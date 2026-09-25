from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gutsignal.features import extract_features, load_and_preprocess

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "artifacts" / "model" / "baseline.joblib"
METRICS_PATH = ROOT / "artifacts" / "model" / "metrics.json"
EVIDENCE_IMAGE_PATH = ROOT / "artifacts" / "report" / "model_evidence.png"
DEMO_SAMPLES_PATH = ROOT / "demo_samples"


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
    if probability < 0.35:
        return (
            "Lower model evidence",
            "The baseline found less evidence of an annotated bowel-sound event in this clip.",
        )
    if probability < 0.65:
        return (
            "Uncertain",
            "The result sits near the decision boundary and should not be treated as a confident detection.",
        )
    return (
        "Higher model evidence",
        "The baseline found more evidence of a sound pattern resembling expert-annotated events in the public dataset.",
    )


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

demo_tab, evidence_tab, method_tab, limits_tab = st.tabs(
    ["Try the prototype", "Evidence", "How it works", "Limits & next steps"]
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
    selected_path: Path | None = None
    example_label: str | None = None

    if mode == "Curated synthetic example":
        manifest = load_demo_manifest()
        example_name = st.selectbox("Example", list(manifest))
        example = manifest[example_name]
        selected_path = DEMO_SAMPLES_PATH / example["file"]
        example_label = example["example_label"]
        st.caption(example["purpose"])
        st.audio(selected_path.read_bytes(), format="audio/wav")
    else:
        uploaded = st.file_uploader("Upload a two-second WAV recording", type=["wav"])
        st.caption("Files are processed only for this demonstration session.")
        if uploaded is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
                temporary.write(uploaded.getbuffer())
                selected_path = Path(temporary.name)

    if selected_path is not None:
        signal, sample_rate = load_and_preprocess(selected_path)
        feature_row = pd.DataFrame([extract_features(selected_path)])
        bundle = load_model_bundle()
        probability = float(bundle["pipeline"].predict_proba(feature_row)[0, 1])

        label, explanation = interpretation(probability)
        left, right = st.columns([1, 2])
        left.metric("Model evidence", f"{probability:.1%}")
        right.markdown(f"### {label}\n{explanation}")
        if example_label is not None:
            st.caption(f"Synthetic pattern description: **{example_label}**")

        st.plotly_chart(waveform_chart(signal, sample_rate), width="stretch")
        st.plotly_chart(spectrogram_chart(signal, sample_rate), width="stretch")
        st.caption(
            "The probability is a research-model output, not a health score. Microphone type, placement, "
            "movement, clothing, speech and background noise may materially affect it."
        )
        if mode == "Upload a WAV" and selected_path.parent == Path(tempfile.gettempdir()):
            selected_path.unlink(missing_ok=True)

with evidence_tab:
    metrics = load_metrics()
    grouped = metrics["participant_grouped_cross_validation"]
    random = metrics["random_recording_cross_validation"]
    uncertainty = metrics["uncertainty_band_evaluation"]

    st.subheader("The less flattering result is the headline result")
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
            f"{uncertainty['uncertain_recordings']:,} of "
            f"{uncertainty['recordings']:,} recordings"
        ),
    )
    uncertainty_columns[1].metric(
        "Accuracy on remaining recordings",
        f"{uncertainty['decided_accuracy']:.1%}",
        delta=(
            f"{(uncertainty['decided_accuracy'] - grouped['accuracy']) * 100:.1f} "
            "percentage points"
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
    st.subheader("A deliberately simple, auditable baseline")
    st.write(
        "The goal was not to maximise a leaderboard score. It was to build a reproducible first "
        "system whose assumptions, failure modes and next experiments could be inspected."
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
        - A small public dataset does not justify pretending complexity guarantees generalisation.
        - Interpretable features make errors and shortcuts easier to investigate.
        - A transparent baseline creates a reference point for any later deep-learning model.
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
        2. Stress-test motion, speech, clothing friction and imperfect sensor contact.
        3. Compare performance across devices and skin-placement variation.
        4. Confirm the uncertainty range on an external participant holdout and define when the model
           must withhold a result.
        5. Co-design explanations with users and clinicians before exposing outputs.
        """
    )

    st.subheader("User research in progress")
    st.write(
        "A short questionnaire and follow-up interviews are testing a separate product question: "
        "what would make longitudinal gut-signal insights feel useful, trustworthy and appropriately "
        "uncertain to a non-clinical user? Findings will be added as evidence, not retrofitted to "
        "support a predetermined conclusion."
    )
    research_columns = st.columns(3)
    research_columns[0].markdown(
        "**Usefulness**\n\nWhich decisions or reflections would an insight genuinely support?"
    )
    research_columns[1].markdown(
        "**Trust**\n\nWhat explanation and evidence would make an output credible?"
    )
    research_columns[2].markdown(
        "**Boundaries**\n\nWhere should a consumer product abstain or recommend clinical advice?"
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
    '<div class="footer-note">Independent application proof of work. '
    "Public data only · non-diagnostic · September 2026</div>",
    unsafe_allow_html=True,
)
