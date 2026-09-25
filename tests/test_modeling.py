import numpy as np

from gutsignal.modeling import probability_metrics


def test_probability_metrics_has_expected_confusion_matrix() -> None:
    target = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.8, 0.4, 0.9])
    metrics = probability_metrics(target, probability)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["accuracy"] == 0.5
