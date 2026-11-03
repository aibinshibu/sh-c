"""Small helpers shared by the video tests (not collected as tests)."""
from __future__ import annotations

import time
from pathlib import Path

TERMINAL = {"Completed", "Failed", "Cancelled"}


def upload(client, video_bytes: bytes, filename: str, headers: dict) -> str:
    res = client.post("/video-jobs", headers=headers,
                      files={"file": (filename, video_bytes, "video/mp4")})
    assert res.status_code == 200, f"upload failed: {res.text}"
    return res.json()["job_id"]


def poll_until(client, job_id: str, headers: dict, statuses=TERMINAL, timeout=180.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        res = client.get(f"/video-jobs/{job_id}", headers=headers)
        assert res.status_code == 200, res.text
        data = res.json()
        if data["status"] in statuses:
            return data
        time.sleep(0.2)
    raise TimeoutError(f"job {job_id} never reached {statuses} in {timeout}s")


def run_video(client, path: Path, headers: dict) -> dict:
    """Upload a video, wait for completion and return {product: quantity}."""
    job = poll_until(client, upload(client, Path(path).read_bytes(), Path(path).name, headers), headers)
    assert job["status"] == "Completed", f"job did not complete: {job.get('error')}"
    return {i["product"]: i["quantity"] for i in job.get("items", [])}
