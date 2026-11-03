"""Shared fixtures for the Smart Shopping Cart backend tests.

Design rules
------------
* The real ``supermarket1.db`` is NEVER modified: every test runs against a
  throw-away copy (``tmp_db``), which is activated automatically.
* Paths are resolved relative to the project, not hard-coded to a drive letter.
  Set CART_TEST_VIDEOS to use a different folder of test videos.
* Gemini and Razorpay are never called.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(BACKEND_DIR / ".env")
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("GOOGLE_API_KEY", "dummy-key-not-used-in-tests")
os.environ.setdefault("MAX_CONCURRENT_VIDEOS", "2")

VIDEO_DIR = Path(os.getenv("CART_TEST_VIDEOS", BACKEND_DIR.parent / "test_videos"))

# Ground truth, verified by inspecting frames of the clips.
# Keys are CATALOGUE product names (the chess board's product is 'chessboard').
EXPECTED_VIDEO_A = {"book": 1}
EXPECTED_VIDEO_B = {"book": 1, "chessboard": 1}


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Create a private seeded database for every test."""
    import db
    from setupdb import setup_database

    db_copy = tmp_path / "test.db"
    setup_database(db_copy)
    monkeypatch.setattr(db, "DB_PATH", db_copy)
    return db_copy


@pytest.fixture()
def api_key() -> str:
    return os.environ["API_KEY"]


@pytest.fixture()
def headers(api_key) -> dict:
    return {"X-API-Key": api_key}


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from main import app

    return TestClient(app)


@pytest.fixture(scope="session")
def models_loaded():
    """Compile the OpenVINO model pool once for all video tests."""
    from main import manager

    manager.load_models()
    return manager


def video_path(name: str) -> Path:
    path = VIDEO_DIR / name
    if not path.is_file():
        pytest.skip(f"test video not found: {path} (set CART_TEST_VIDEOS)")
    return path
