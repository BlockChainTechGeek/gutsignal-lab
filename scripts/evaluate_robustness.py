from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib

from gutsignal.audio import decode_wav
from gutsignal.robustness import stress_matrix

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    model_path = ROOT / "artifacts/model/baseline.joblib"
    manifest = json.loads((ROOT / "demo_samples/manifest.json").read_text())
    clips = {
        name: decode_wav((ROOT / "demo_samples" / item["file"]).read_bytes())
        for name, item in manifest.items()
    }
    rows = stress_matrix(clips, joblib.load(model_path))
    report = {
        "scope": "Generated non-human audio: engineering sensitivity checks, not accuracy evidence",
        "seed": 42,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "input_sha256": {
            name: hashlib.sha256((ROOT / "demo_samples" / item["file"]).read_bytes()).hexdigest()
            for name, item in manifest.items()
        },
        "summary": {
            "cases": len(rows),
            "withheld": sum(not row["quality_accepted"] for row in rows),
            "accepted_state_changes": sum(
                row["quality_accepted"] and row["state_changed"] for row in rows
            ),
            "max_accepted_absolute_shift_percentage_points": max(
                abs(row["change_percentage_points"])
                for row in rows
                if row["change_percentage_points"] is not None
            ),
        },
        "cases": rows,
    }
    destination = ROOT / "artifacts/report/robustness.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
