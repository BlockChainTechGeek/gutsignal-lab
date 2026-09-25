from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

FILENAME_PATTERN = re.compile(r"^(?P<segment>\d+)_(?P<participant>[a-z]+)$")


@dataclass(frozen=True)
class Recording:
    recording_id: str
    participant_id: str
    segment_id: int
    wav_path: Path
    annotation_path: Path
    has_event: bool
    event_count: int


def parse_recording_stem(stem: str) -> tuple[int, str]:
    match = FILENAME_PATTERN.fullmatch(stem)
    if not match:
        raise ValueError(f"Unexpected recording filename: {stem}")
    return int(match.group("segment")), match.group("participant")


def annotation_event_count(path: Path) -> int:
    annotations = pd.read_csv(path)
    required = {"start", "end", "fmin", "fmax", "category"}
    missing = required.difference(annotations.columns)
    if missing:
        raise ValueError(f"{path.name} is missing annotation columns: {sorted(missing)}")
    return len(annotations)


def discover_recordings(data_dir: Path) -> list[Recording]:
    recordings: list[Recording] = []
    for wav_path in sorted(data_dir.glob("*.wav")):
        annotation_path = wav_path.with_suffix(".csv")
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing annotation file for {wav_path.name}")
        segment_id, participant_id = parse_recording_stem(wav_path.stem)
        event_count = annotation_event_count(annotation_path)
        recordings.append(
            Recording(
                recording_id=wav_path.stem,
                participant_id=participant_id,
                segment_id=segment_id,
                wav_path=wav_path,
                annotation_path=annotation_path,
                has_event=event_count > 0,
                event_count=event_count,
            )
        )
    if not recordings:
        raise FileNotFoundError(f"No WAV recordings found in {data_dir}")
    return recordings


def recording_manifest(recordings: list[Recording]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "recording_id": item.recording_id,
                "participant_id": item.participant_id,
                "segment_id": item.segment_id,
                "wav_path": str(item.wav_path),
                "annotation_path": str(item.annotation_path),
                "has_event": int(item.has_event),
                "event_count": item.event_count,
            }
            for item in recordings
        ]
    )
