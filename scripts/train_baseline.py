from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

from gutsignal.modeling import build_baseline, probability_metrics, uncertainty_band_metrics

METADATA_COLUMNS = {"recording_id", "participant_id", "has_event", "event_count"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.features)
    feature_columns = [column for column in frame.columns if column not in METADATA_COLUMNS]
    target = frame["has_event"].to_numpy()
    groups = frame["participant_id"].to_numpy()

    participant_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    random_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grouped_model = build_baseline(feature_columns)
    grouped_probability = cross_val_predict(
        grouped_model,
        frame,
        target,
        groups=groups,
        cv=participant_cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    random_model = build_baseline(feature_columns)
    random_probability = cross_val_predict(
        random_model,
        frame,
        target,
        cv=random_cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    metrics = {
        "dataset": {
            "recordings": len(frame),
            "participant_groups": int(frame["participant_id"].nunique()),
            "event_positive": int(target.sum()),
            "event_negative": int((1 - target).sum()),
        },
        "participant_grouped_cross_validation": probability_metrics(target, grouped_probability),
        "random_recording_cross_validation": probability_metrics(target, random_probability),
        "uncertainty_band_evaluation": uncertainty_band_metrics(target, grouped_probability),
        "interpretation": (
            "The grouped estimate is the headline result. The random-recording estimate is included "
            "only to make the generalisation gap visible; recordings from one participant should not "
            "appear in both training and evaluation folds."
        ),
    }

    participant_map = {
        participant_id: f"group_{index:02d}"
        for index, participant_id in enumerate(sorted(frame["participant_id"].unique()), start=1)
    }
    predictions = pd.DataFrame(
        {
            "recording_id": [f"sample_{index:04d}" for index in range(1, len(frame) + 1)],
            "participant_id": frame["participant_id"].map(participant_map),
            "has_event": frame["has_event"],
        }
    )
    predictions["participant_grouped_probability"] = grouped_probability
    predictions["random_recording_probability"] = random_probability

    final_model = build_baseline(feature_columns)
    final_model.fit(frame, target)

    classifier = final_model.named_steps["classifier"]
    coefficient_frame = pd.DataFrame(
        {
            "feature": final_model.named_steps["features"].get_feature_names_out(),
            "coefficient": classifier.coef_[0],
        }
    ).assign(abs_coefficient=lambda data: np.abs(data["coefficient"]))
    coefficient_frame = coefficient_frame.sort_values("abs_coefficient", ascending=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    coefficient_frame.to_csv(args.output_dir / "feature_coefficients.csv", index=False)
    joblib.dump(
        {
            "pipeline": final_model,
            "feature_columns": feature_columns,
            "metadata": {
                "purpose": "research demonstration only",
                "target": "presence of an expert-annotated bowel-sound event in a two-second clip",
                "sample_rate_hz": 8_000,
                "threshold": 0.5,
            },
        },
        args.output_dir / "baseline.joblib",
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
