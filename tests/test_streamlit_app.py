from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_public_demo_renders_without_exceptions() -> None:
    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=30)
    app.run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Try the prototype",
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

    app.selectbox[0].select("Synthetic boundary pattern").run()
    assert not app.exception
    assert app.metric[0].value.endswith("%")
    assert any(markdown.value.startswith("### Uncertain") for markdown in app.markdown)
