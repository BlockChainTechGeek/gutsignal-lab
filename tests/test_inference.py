from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from gutsignal.audio import AudioClip, decode_wav
from gutsignal.features import (
    extract_features,
    features_from_preprocessed,
    preprocess_signal,
)
from gutsignal.inference import evidence_state, predict_clip

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def bundle() -> dict:
    return joblib.load(ROOT / "artifacts/model/baseline.joblib")


@pytest.mark.parametrize("name", ["event", "nonevent", "boundary"])
def test_in_memory_inference_matches_original_path(name: str, bundle: dict) -> None:
    path = ROOT / f"demo_samples/synthetic_{name}.wav"
    clip = decode_wav(path.read_bytes())
    result = predict_clip(clip, bundle)
    original = bundle["pipeline"].predict_proba(pd.DataFrame([extract_features(path)]))[0, 1]
    assert result.probability == pytest.approx(original, abs=1e-5)


def test_invalid_quality_never_calls_model() -> None:
    # Empty bundle would fail if feature extraction or model invocation were attempted.
    result = predict_clip(AudioClip(np.zeros(16000), 8000), {})
    assert result.probability is None
    assert result.state == "Withheld: input quality"


def test_model_schema_mismatch_fails_closed(bundle: dict) -> None:
    clip = decode_wav((ROOT / "demo_samples/synthetic_event.wav").read_bytes())
    wrong = dict(bundle, feature_columns=["missing_feature"])
    with pytest.raises(ValueError, match="schema"):
        predict_clip(clip, wrong)


def test_features_are_finite_and_do_not_mutate_input() -> None:
    clip = decode_wav((ROOT / "demo_samples/synthetic_event.wav").read_bytes())
    original = clip.signal.copy()
    signal, rate = preprocess_signal(clip.signal, clip.sample_rate)
    features = features_from_preprocessed(signal, rate)
    np.testing.assert_array_equal(clip.signal, original)
    assert all(np.isfinite(list(features.values())))
    assert len(features) == 51


@pytest.mark.parametrize("bad", [np.zeros(1), np.full(16000, np.nan)])
def test_invalid_preprocessing_input_fails_cleanly(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        preprocess_signal(bad, 8000)


def test_resampled_inference_matches_file_path(tmp_path: Path, bundle: dict) -> None:
    clip = decode_wav((ROOT / "demo_samples/synthetic_event.wav").read_bytes())
    high_rate = librosa.resample(clip.signal, orig_sr=8000, target_sr=44100)
    path = tmp_path / "generated_44100.wav"
    sf.write(path, high_rate * 0.5, 44100, subtype="FLOAT")
    result = predict_clip(decode_wav(path.read_bytes()), bundle)
    original = bundle["pipeline"].predict_proba(pd.DataFrame([extract_features(path)]))[0, 1]
    assert result.probability == pytest.approx(original, abs=1e-5)


@pytest.mark.parametrize(
    ("probability", "state"),
    [
        (0.3499, "Lower model evidence"),
        (0.35, "Uncertain"),
        (0.6499, "Uncertain"),
        (0.65, "Higher model evidence"),
    ],
)
def test_evidence_boundaries(probability: float, state: str) -> None:
    assert evidence_state(probability) == state


@pytest.mark.parametrize("probability", [float("nan"), -0.1, 1.1])
def test_invalid_model_scores_are_rejected(probability: float) -> None:
    with pytest.raises(ValueError):
        evidence_state(probability)
