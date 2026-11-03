"""Test 2: the same video analysed twice at once must give identical, correct results."""
import threading

import pytest
from fastapi.testclient import TestClient

from conftest import EXPECTED_VIDEO_A, video_path
from helpers import run_video

pytestmark = pytest.mark.slow


def test_same_video_twice_at_once(models_loaded, headers):
    from main import app

    path = video_path("video_a.mp4")
    out, barrier = {}, threading.Barrier(2)

    def worker(label):
        c = TestClient(app)
        barrier.wait()
        out[label] = run_video(c, path, headers)

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("run_1", "run_2")]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert out["run_1"] == out["run_2"], "two simultaneous runs disagree"
    assert out["run_1"] == EXPECTED_VIDEO_A, "result differs from ground truth"
