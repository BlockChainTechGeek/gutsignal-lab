import numpy as np

from gutsignal.modeling import probability_metrics, uncertainty_band_metrics


def test_probability_metrics_has_expected_confusion_matrix() -> None:
    target = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.8, 0.4, 0.9])
    metrics = probability_metrics(target, probability)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["accuracy"] == 0.5


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
