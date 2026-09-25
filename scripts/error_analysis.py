from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/report"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = json.loads((args.model_dir / "metrics.json").read_text())
    predictions = pd.read_csv(args.model_dir / "oof_predictions.csv")
    probability_column = "participant_grouped_probability"
    predictions["prediction"] = (predictions[probability_column] >= 0.5).astype(int)
    predictions["correct"] = predictions["prediction"] == predictions["has_event"]
    predictions["outcome"] = np.select(
        [
            (predictions["has_event"] == 1) & (predictions["prediction"] == 1),
            (predictions["has_event"] == 0) & (predictions["prediction"] == 0),
            (predictions["has_event"] == 0) & (predictions["prediction"] == 1),
        ],
        ["true_positive", "true_negative", "false_positive"],
        default="false_negative",
    )
    predictions["error_confidence"] = np.where(
        predictions["prediction"] == 1,
        predictions[probability_column],
        1 - predictions[probability_column],
    )

    participant = (
        predictions.groupby("participant_id")
        .agg(
            recordings=("recording_id", "count"),
            event_rate=("has_event", "mean"),
            accuracy=("correct", "mean"),
            mean_probability=(probability_column, "mean"),
            errors=("correct", lambda values: int((~values).sum())),
        )
        .reset_index()
    )
    participant["error_rate"] = 1 - participant["accuracy"]

    confident_errors = (
        predictions.loc[~predictions["correct"]]
        .sort_values("error_confidence", ascending=False)
        .head(25)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    participant.to_csv(args.output_dir / "participant_metrics.csv", index=False)
    confident_errors.to_csv(args.output_dir / "most_confident_errors.csv", index=False)

    grouped = metrics["participant_grouped_cross_validation"]
    random = metrics["random_recording_cross_validation"]
    metric_names = ["average_precision", "roc_auc", "f1"]
    labels = ["Average precision", "ROC-AUC", "F1"]
    x = np.arange(len(labels))
    width = 0.34

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    axes[0].bar(x - width / 2, [grouped[name] for name in metric_names], width, label="Grouped")
    axes[0].bar(x + width / 2, [random[name] for name in metric_names], width, label="Random")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.7, 1.0)
    axes[0].set_title("Validation design changes the result")
    axes[0].set_ylabel("Score")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    confusion = np.asarray(grouped["confusion_matrix"])
    image = axes[1].imshow(confusion, cmap="Oranges")
    for row in range(2):
        for column in range(2):
            axes[1].text(column, row, f"{confusion[row, column]:,}", ha="center", va="center")
    axes[1].set_xticks([0, 1], ["No event", "Event"])
    axes[1].set_yticks([0, 1], ["No event", "Event"])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Expert annotation")
    axes[1].set_title("Grouped out-of-fold decisions")
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)

    observed, predicted = calibration_curve(
        predictions["has_event"], predictions[probability_column], n_bins=8, strategy="quantile"
    )
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Ideal")
    axes[2].plot(predicted, observed, marker="o", color="#d95f45", label="Baseline")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Observed event rate")
    axes[2].set_title("Probability calibration")
    axes[2].grid(alpha=0.2)
    axes[2].legend(frameon=False)

    figure.suptitle("GutSignal baseline: evidence before polish", fontsize=15, fontweight="bold")
    figure.tight_layout()
    figure.savefig(args.output_dir / "model_evidence.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(participant.sort_values("error_rate", ascending=False).to_string(index=False))
    print(f"\nSaved analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
