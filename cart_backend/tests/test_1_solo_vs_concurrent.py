"""Test 1: correctness against ground truth, and solo == concurrent.

A detector can be consistently wrong, so every run is compared with the known
content of the clip, not only with another run.
"""
import threading

import pytest
from fastapi.testclient import TestClient

from conftest import EXPECTED_VIDEO_A, EXPECTED_VIDEO_B, video_path
from helpers import run_video

pytestmark = pytest.mark.slow


def test_solo_results_match_ground_truth(models_loaded, client, headers):
    assert run_video(client, video_path("video_a.mp4"), headers) == EXPECTED_VIDEO_A
    assert run_video(client, video_path("video_b.mp4"), headers) == EXPECTED_VIDEO_B


def test_concurrent_results_equal_ground_truth(models_loaded, headers):
    from main import app

    a, b = video_path("video_a.mp4"), video_path("video_b.mp4")
    out, barrier = {}, threading.Barrier(2)

    def worker(label, path):
        c = TestClient(app)
        barrier.wait()
        out[label] = run_video(c, path, headers)

    threads = [threading.Thread(target=worker, args=("A", a)),
               threading.Thread(target=worker, args=("B", b))]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert out["A"] == EXPECTED_VIDEO_A, "video A changed when run concurrently"
    assert out["B"] == EXPECTED_VIDEO_B, "video B changed when run concurrently"
