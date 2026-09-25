from pathlib import Path

import pandas as pd

from gutsignal.dataset import annotation_event_count, parse_recording_stem


def test_parse_recording_stem() -> None:
    assert parse_recording_stem("320_c") == (320, "c")


def test_empty_annotation_has_zero_events(tmp_path: Path) -> None:
    annotation = tmp_path / "sample.csv"
    pd.DataFrame(columns=["start", "end", "fmin", "fmax", "category"]).to_csv(
        annotation, index=False
    )
    assert annotation_event_count(annotation) == 0
