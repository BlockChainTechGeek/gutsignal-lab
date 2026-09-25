from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_baseline(feature_columns: list[str]) -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [("numeric", numeric, feature_columns)],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            ),
            ("classifier", LogisticRegression(C=0.3, max_iter=3_000, random_state=42)),
        ]
    )


def probability_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = (probability >= 0.5).astype(int)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "average_precision": average_precision_score(y_true, probability),
        "roc_auc": roc_auc_score(y_true, probability),
        "brier_score": brier_score_loss(y_true, probability),
        "confusion_matrix": matrix.tolist(),
        "threshold": 0.5,
    }


def uncertainty_band_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    lower: float = 0.35,
    upper: float = 0.65,
) -> dict[str, Any]:
    if not 0 <= lower < 0.5 < upper <= 1:
        raise ValueError("Expected 0 <= lower < 0.5 < upper <= 1")

    prediction = (probability >= 0.5).astype(int)
    uncertain = (probability >= lower) & (probability < upper)
    decided = ~uncertain
    errors = prediction != y_true

    if not decided.any():
        raise ValueError("The uncertainty band leaves no decided recordings")

    total_errors = int(errors.sum())
    errors_inside_band = int((errors & uncertain).sum())
    decided_target = y_true[decided]
    decided_prediction = prediction[decided]

    return {
        "lower_bound": lower,
        "upper_bound": upper,
        "recordings": len(y_true),
        "uncertain_recordings": int(uncertain.sum()),
        "uncertain_fraction": float(uncertain.mean()),
        "decision_coverage": float(decided.mean()),
        "decided_accuracy": accuracy_score(decided_target, decided_prediction),
        "decided_balanced_accuracy": balanced_accuracy_score(
            decided_target, decided_prediction
        ),
        "decided_precision": precision_score(
            decided_target, decided_prediction, zero_division=0
        ),
        "decided_recall": recall_score(decided_target, decided_prediction, zero_division=0),
        "errors_inside_band": errors_inside_band,
        "error_capture_rate": errors_inside_band / total_errors if total_errors else 0.0,
        "forced_error_rate_inside_band": float(errors[uncertain].mean())
        if uncertain.any()
        else 0.0,
    }
