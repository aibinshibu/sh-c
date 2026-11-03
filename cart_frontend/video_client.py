"""Frontend video client: the same three methods the page used on the old VideoCartManager.

    start_job(video_bytes, filename) -> job_id
    cancel_job(job_id)
    snapshot(job_id) -> dict | None

The real work (OpenVINO model, tracking, price lookup) now happens in the backend.
"""

from __future__ import annotations

import api_client


def _failed_state(job_id: str, message: str) -> dict:
    """What the page shows if the backend cannot be reached while it is polling."""
    return {
        "job_id": job_id, "filename": "", "status": "Failed", "progress": 0.0,
        "processed_frames": 0, "total_frames": 0, "items": [], "total": 0.0,
        "unmatched_labels": [], "price_queries": {}, "events": [], "unresolved": [],
        "error": message, "cancel_requested": False,
    }


class VideoCartClient:
    def start_job(self, video_bytes: bytes, filename: str) -> str:
        reply = api_client.request(
            "POST", "/video-jobs",
            files={"file": (filename, video_bytes)},
            timeout=api_client.UPLOAD_TIMEOUT,
        )
        return reply["job_id"]

    def cancel_job(self, job_id: str | None) -> None:
        if not job_id:
            return
        try:
            api_client.request("DELETE", f"/video-jobs/{job_id}")
        except Exception:
            pass  # nothing useful to do if the old job cannot be cancelled

    def snapshot(self, job_id: str | None) -> dict | None:
        if not job_id:
            return None
        try:
            return api_client.request("GET", f"/video-jobs/{job_id}", allow_404=True)
        except Exception as error:  # the page polls every second, so never let this crash it
            return _failed_state(job_id, str(error))
