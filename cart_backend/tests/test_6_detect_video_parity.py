"""Test 6: the command-line detector and the web pipeline give the same final cart."""
import json
import subprocess
import sys

import cv2
import pytest

from cart_detector import CartAnalyzer, OpenVinoModel, Settings
from conftest import BACKEND_DIR, EXPECTED_VIDEO_A, video_path

pytestmark = pytest.mark.slow


def test_cli_matches_analyzer(tmp_path):
    video = video_path("video_a.mp4")
    model_path = BACKEND_DIR / "models" / "best.xml"
    out = tmp_path / "detect_out.mp4"

    proc = subprocess.run(
        [sys.executable, str(BACKEND_DIR / "detect_video.py"), "--model", str(model_path),
         "--video", str(video), "--output", str(out), "--no-preview"],
        cwd=str(BACKEND_DIR), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    cli_cart = json.loads(out.with_suffix(".events.json").read_text()).get("final_cart", {})

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    analyzer = CartAnalyzer(OpenVinoModel(model_path), Settings(), fps)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        analyzer.process(frame, idx)
        idx += 1
    cap.release()
    _, web_cart = analyzer.finish()

    assert cli_cart == web_cart
    assert web_cart == {"book": 1}, "final cart differs from ground truth"
