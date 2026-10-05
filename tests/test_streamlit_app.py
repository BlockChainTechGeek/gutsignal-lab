from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from gutsignal.audio import AudioClip, encode_wav


def test_public_demo_renders_without_exceptions() -> None:
    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=30)
    app.run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Try the prototype",
        "Robustness lab",
        "Evidence",
        "How it works",
        "Limits & next steps",
    ]
    assert app.title[0].value == "GutSignal Lab"
    assert app.metric[0].value.endswith("%")
    assert any(
        metric.label == "Marked uncertain" and metric.value == "13.5%" for metric in app.metric
    )
    assert any(
        metric.label == "Accuracy on remaining recordings" and metric.value == "86.9%"
        for metric in app.metric
    )
    assert any(
        "What four exploratory conversations revealed" in subheader.value
        for subheader in app.subheader
    )
    assert any("Always predicting event gives F1 0.888" in caption.value for caption in app.caption)

    app.selectbox[0].select("Synthetic boundary pattern").run()
    assert not app.exception
    assert app.metric[0].value.endswith("%")
    assert any(markdown.value.startswith("### Uncertain") for markdown in app.markdown)


def test_workbench_withholds_silence_without_crashing() -> None:
    app = AppTest.from_file(Path(__file__).parents[1] / "streamlit_app.py", default_timeout=30)
    app.run()
    app.selectbox[2].select("Silence").run()
    assert not app.exception
    assert any(
        metric.label == "Changed score" and metric.value == "Withheld" for metric in app.metric
    )
    assert any("No usable variation" in warning.value for warning in app.warning)


def test_upload_mode_renders_without_input() -> None:
    app = AppTest.from_file(Path(__file__).parents[1] / "streamlit_app.py", default_timeout=30)
    app.run()
    app.radio[0].set_value("Upload a WAV").run()
    assert not app.exception
    assert not any(metric.label == "Model evidence" for metric in app.metric)


@pytest.mark.parametrize("payload", [b"broken wav", encode_wav(AudioClip(np.zeros(16000), 8000))])
def test_bad_upload_shows_warning_without_a_score(payload: bytes, monkeypatch) -> None:
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: BytesIO(payload))
    app = AppTest.from_file(Path(__file__).parents[1] / "streamlit_app.py", default_timeout=30)
    app.run()
    app.radio[0].set_value("Upload a WAV").run()
    assert not app.exception
    assert app.warning
    assert not any(metric.label == "Model evidence" for metric in app.metric)
