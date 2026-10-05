import numpy as np
import pytest

from gutsignal.modeling import (
    class_balance_reference,
    probability_metrics,
    uncertainty_band_metrics,
)


def test_probability_metrics_has_expected_confusion_matrix() -> None:
    target = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.8, 0.4, 0.9])
    metrics = probability_metrics(target, probability)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["accuracy"] == 0.5


def test_class_balance_reference_matches_constant_predictions() -> None:
    target = np.array([0, 0, 1, 1, 1, 1])
    actual = probability_metrics(target, np.ones(6))
    reference = class_balance_reference(4, 2)
    for metric in ["accuracy", "balanced_accuracy", "f1"]:
        assert reference[metric] == pytest.approx(actual[metric])
    assert reference["non_event_recall"] == 0.0


@pytest.mark.parametrize("positive,negative", [(0, 2), (4, 0), (-1, 2)])
def test_class_balance_reference_requires_both_classes(positive: int, negative: int) -> None:
    with pytest.raises(ValueError):
        class_balance_reference(positive, negative)


def test_uncertainty_band_metrics_reports_coverage_and_error_capture() -> None:
    target = np.array([0, 1, 0, 1])
    probability = np.array([0.1, 0.4, 0.6, 0.9])

    metrics = uncertainty_band_metrics(target, probability)

    assert metrics["uncertain_recordings"] == 2
    assert metrics["uncertain_fraction"] == 0.5
    assert metrics["decision_coverage"] == 0.5
    assert metrics["decided_accuracy"] == 1.0
    assert metrics["errors_inside_band"] == 2
    assert metrics["error_capture_rate"] == 1.0
