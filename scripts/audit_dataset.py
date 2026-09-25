from __future__ import annotations

import argparse
from pathlib import Path

from gutsignal.dataset import discover_recordings, recording_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/manifest.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = recording_manifest(discover_recordings(args.data_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)

    participant_summary = (
        manifest.groupby("participant_id")
        .agg(recordings=("recording_id", "count"), events=("has_event", "sum"))
        .assign(event_rate=lambda frame: frame["events"] / frame["recordings"])
    )
    print(f"Recordings: {len(manifest):,}")
    print(f"Participants/groups: {manifest['participant_id'].nunique()}")
    print(f"Event-positive files: {manifest['has_event'].sum():,}")
    print(f"Event-negative files: {(1 - manifest['has_event']).sum():,}")
    print("\nPer-group distribution:\n")
    print(participant_summary.to_string(float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
