from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 8_000
DURATION_SECONDS = 2.0

EXAMPLES = {
    "Synthetic event-like pattern": {
        "file": "synthetic_event.wav",
        "example_label": "event-like synthetic pattern",
        "purpose": "Illustrates how a stronger transient pattern moves through the pipeline.",
        "bursts": [(0.34, 180, 0.55), (0.92, 260, 0.65), (1.47, 140, 0.50)],
        "noise": 0.012,
    },
    "Synthetic low-activity pattern": {
        "file": "synthetic_nonevent.wav",
        "example_label": "low-activity synthetic pattern",
        "purpose": "Illustrates a quieter signal with no human recording attached.",
        "bursts": [(1.12, 110, 0.10)],
        "noise": 0.006,
    },
    "Synthetic boundary pattern": {
        "file": "synthetic_boundary.wav",
        "example_label": "ambiguous synthetic pattern",
        "purpose": "Illustrates why the interface includes an uncertainty state.",
        "bursts": [(0.58, 160, 0.22), (1.31, 220, 0.18)],
        "noise": 0.010,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("demo_samples"))
    return parser.parse_args()


def synthetic_signal(
    bursts: list[tuple[float, float, float]], noise: float, rng: np.random.Generator
) -> np.ndarray:
    time = np.arange(int(SAMPLE_RATE * DURATION_SECONDS)) / SAMPLE_RATE
    signal = rng.normal(0.0, noise, size=time.size)
    for centre, frequency, amplitude in bursts:
        envelope = np.exp(-0.5 * ((time - centre) / 0.045) ** 2)
        signal += amplitude * envelope * np.sin(2 * np.pi * frequency * time)
    peak = np.max(np.abs(signal))
    return (signal / peak if peak else signal).astype(np.float32)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    manifest = {}
    for label, details in EXAMPLES.items():
        destination = args.output_dir / details["file"]
        signal = synthetic_signal(details["bursts"], details["noise"], rng)
        sf.write(destination, signal, SAMPLE_RATE, subtype="PCM_16")
        manifest[label] = {
            "file": details["file"],
            "example_label": details["example_label"],
            "purpose": details["purpose"],
        }

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
