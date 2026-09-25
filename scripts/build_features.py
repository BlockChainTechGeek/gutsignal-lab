from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from gutsignal.dataset import discover_recordings
from gutsignal.features import extract_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/features.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for recording in tqdm(discover_recordings(args.data_dir), desc="Extracting features"):
        row = {
            "recording_id": recording.recording_id,
            "participant_id": recording.participant_id,
            "has_event": int(recording.has_event),
            "event_count": recording.event_count,
        }
        row.update(extract_features(recording.wav_path))
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
