import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from gutsignal.audio import check_quality, decode_wav
from gutsignal.inference import predict_clip
from gutsignal.robustness import STRESS_CASES, perturb, stress_matrix

ROOT = Path(__file__).parents[1]


def clip_and_bundle() -> tuple:
    clip = decode_wav((ROOT / "demo_samples/synthetic_event.wav").read_bytes())
    return clip, joblib.load(ROOT / "artifacts/model/baseline.joblib")


def test_noise_is_deterministic_and_does_not_mutate_source() -> None:
    clip, _ = clip_and_bundle()
    before = clip.signal.copy()
    first = perturb(clip, "Noise", 10)
    np.testing.assert_array_equal(first.signal, perturb(clip, "Noise", 10).signal)
    np.testing.assert_array_equal(clip.signal, before)
    assert not np.array_equal(first.signal, before)


def test_peak_normalisation_makes_gain_nearly_invariant() -> None:
    clip, bundle = clip_and_bundle()
    original = predict_clip(clip, bundle)
    changed = predict_clip(perturb(clip, "Gain", -30), bundle)
    assert changed.probability == pytest.approx(original.probability, abs=1e-5)


@pytest.mark.parametrize(
    "kind,level",
    [("Silence", 0), ("Constant DC", 0), ("Clipping", 8), ("Dropout", 25), ("Dropout", 100)],
)
def test_known_stresses_are_withheld(kind: str, level: float) -> None:
    clip, bundle = clip_and_bundle()
    changed = perturb(clip, kind, level)
    assert not check_quality(changed).accepted
    assert predict_clip(changed, bundle).probability is None


@pytest.mark.parametrize(
    "kind,level", [("Noise", -1), ("Gain", 10), ("Clipping", 0), ("Dropout", 101), ("Unknown", 0)]
)
def test_invalid_perturbations_fail_cleanly(kind: str, level: float) -> None:
    clip, _ = clip_and_bundle()
    with pytest.raises(ValueError):
        perturb(clip, kind, level)


def test_report_matches_current_implementation() -> None:
    report = json.loads((ROOT / "artifacts/report/robustness.json").read_text())
    assert (
        report["model_sha256"]
        == hashlib.sha256((ROOT / "artifacts/model/baseline.joblib").read_bytes()).hexdigest()
    )
    manifest = json.loads((ROOT / "demo_samples/manifest.json").read_text())
    clips = {
        name: decode_wav((ROOT / "demo_samples" / item["file"]).read_bytes())
        for name, item in manifest.items()
    }
    for name, item in manifest.items():
        assert (
            report["input_sha256"][name]
            == hashlib.sha256((ROOT / "demo_samples" / item["file"]).read_bytes()).hexdigest()
        )
    _, bundle = clip_and_bundle()
    rows = stress_matrix(clips, bundle)
    assert len(rows) == 3 * len(STRESS_CASES)
    assert len(rows) == report["summary"]["cases"]
    assert sum(not row["quality_accepted"] for row in rows) == report["summary"]["withheld"]
    for expected, actual in zip(report["cases"], rows, strict=True):
        assert expected["state"] == actual["state"]
        assert expected["score"] == pytest.approx(actual["score"], abs=1e-5)
