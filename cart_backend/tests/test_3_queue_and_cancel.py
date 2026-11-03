"""Test 3: queueing, cancellation and model-pool recycling (pool size = 2)."""
import pytest

from conftest import EXPECTED_VIDEO_A, video_path
from helpers import poll_until, upload

pytestmark = pytest.mark.slow


@pytest.fixture()
def vbytes():
    return video_path("video_a.mp4").read_bytes()


def test_third_job_is_queued_then_completes(models_loaded, client, headers, vbytes):
    j1, j2, j3 = (upload(client, vbytes, f"j{i}.mp4", headers) for i in (1, 2, 3))
    assert client.get(f"/video-jobs/{j3}", headers=headers).json()["status"] == "Queued"
    assert client.get("/health").json()["videos_queued"] >= 1
    for jid in (j1, j2, j3):
        assert poll_until(client, jid, headers)["status"] == "Completed"


def test_cancel_queued_job(models_loaded, client, headers, vbytes):
    j4, j5, j6 = (upload(client, vbytes, f"j{i}.mp4", headers) for i in (4, 5, 6))
    assert client.delete(f"/video-jobs/{j6}", headers=headers).status_code == 200
    assert poll_until(client, j6, headers)["status"] == "Cancelled"
    assert poll_until(client, j4, headers)["status"] == "Completed"
    assert poll_until(client, j5, headers)["status"] == "Completed"


def test_cancel_running_job_returns_model_to_pool(models_loaded, client, headers, vbytes):
    j7 = upload(client, vbytes, "j7.mp4", headers)
    poll_until(client, j7, headers, {"Processing"})
    client.delete(f"/video-jobs/{j7}", headers=headers)
    assert poll_until(client, j7, headers)["status"] == "Cancelled"

    # The pool must not be starved: a new job still runs to a correct result.
    j8 = upload(client, vbytes, "j8.mp4", headers)
    done = poll_until(client, j8, headers)
    assert done["status"] == "Completed"
    assert {i["product"]: i["quantity"] for i in done["items"]} == EXPECTED_VIDEO_A


def test_unknown_job_returns_404(client, headers):
    assert client.get("/video-jobs/does-not-exist", headers=headers).status_code == 404
