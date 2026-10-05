from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from gutsignal.audio import AudioClip, QualityCheck, check_quality
from gutsignal.features import features_from_preprocessed, preprocess_signal


@dataclass(frozen=True)
class Prediction:
    quality: QualityCheck
    probability: float | None
    state: str


def evidence_state(probability: float) -> str:
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Expected a finite model score between zero and one")
    if probability < 0.35:
        return "Lower model evidence"
    if probability < 0.65:
        return "Uncertain"
    return "Higher model evidence"


def predict_clip(clip: AudioClip, bundle: dict[str, Any]) -> Prediction:
    quality = check_quality(clip)
    if not quality.accepted:
        return Prediction(quality, None, "Withheld: input quality")
    signal, rate = preprocess_signal(clip.signal, clip.sample_rate)
    features = features_from_preprocessed(signal, rate)
    if set(features) != set(bundle["feature_columns"]):
        raise ValueError("Feature schema does not match the bundled model")
    if not all(np.isfinite(value) for value in features.values()):
        raise ValueError("Feature extraction produced invalid values")
    frame = pd.DataFrame([features], columns=bundle["feature_columns"])
    probability = float(bundle["pipeline"].predict_proba(frame)[0, 1])
    return Prediction(quality, probability, evidence_state(probability))
