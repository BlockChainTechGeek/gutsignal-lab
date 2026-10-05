import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_demo_generation_reproduces_bundled_files(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_demo_samples.py"),
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    bundled = json.loads((ROOT / "demo_samples/manifest.json").read_text())
    generated = json.loads((tmp_path / "manifest.json").read_text())
    assert generated == bundled
    for details in bundled.values():
        name = details["file"]
        assert (tmp_path / name).read_bytes() == (ROOT / "demo_samples" / name).read_bytes()
