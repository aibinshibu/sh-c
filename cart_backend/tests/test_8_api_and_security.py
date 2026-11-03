"""Test 8: API authentication, upload validation, input validation and the SQL guard."""
import time

import pytest

import assistant
import db


# ---- authentication ------------------------------------------------------
def test_health_needs_no_key(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_found"] is True


@pytest.mark.parametrize("hdrs", [{}, {"X-API-Key": "wrong"}])
def test_protected_route_rejects_missing_or_wrong_key(client, hdrs):
    r = client.get("/products", headers=hdrs)
    assert r.status_code == 401
    assert r.json()["detail"] == "Missing or wrong API key."


def test_valid_key_lists_products(client, headers):
    r = client.get("/products", headers=headers)
    assert r.status_code == 200 and len(r.json()) >= 38


# ---- video upload validation ---------------------------------------------
def test_empty_upload_is_rejected(client, headers):
    r = client.post("/video-jobs", headers=headers, files={"file": ("empty.mp4", b"")})
    assert r.status_code == 400
    assert r.json()["detail"] == "The uploaded video is empty."


def test_corrupt_video_fails_with_readable_error(models_loaded, client, headers):
    r = client.post("/video-jobs", headers=headers, files={"file": ("bad.mp4", b"not a video")})
    jid = r.json()["job_id"]
    for _ in range(60):
        job = client.get(f"/video-jobs/{jid}", headers=headers).json()
        if job["status"] in ("Failed", "Completed"):
            break
        time.sleep(0.5)
    assert job["status"] == "Failed"
    assert job["error"] == "The uploaded video could not be opened."


# ---- input validation -----------------------------------------------------
def test_login_with_short_phone_is_rejected(client, headers):
    r = client.post("/login", headers=headers, json={"phone_number": "12", "password": "x"})
    assert r.status_code == 400
    assert "8 to 15 digits" in r.json()["detail"]


def test_login_with_missing_field_is_rejected(client, headers):
    assert client.post("/login", headers=headers, json={"phone_number": "1"}).status_code == 400


def test_registration_validation():
    with pytest.raises(ValueError, match="8 to 15 digits"):
        db.create_customer("X", "123", "secret1")
    with pytest.raises(ValueError, match="at least 6"):
        db.create_customer("X", "9876543210", "abc")
    existing = db.get_customer_summaries()[0]["phone_number"]
    with pytest.raises(ValueError, match="already exists"):
        db.create_customer("X", existing, "secret1")


def test_wrong_password_returns_none_and_good_password_works():
    assert db.verify_login("0000000000", "wrong") is None
    phone = "9000000001"
    cid = db.create_customer("Tester", phone, "secret1")
    assert db.verify_login(phone, "secret1")["id"] == cid
    assert db.verify_login(phone, "secret2") is None


def test_password_is_stored_hashed(tmp_db):
    import sqlite3
    db.create_customer("Tester", "9000000002", "secret1")
    with sqlite3.connect(tmp_db) as c:
        stored = c.execute("SELECT password_hash FROM customers WHERE phone_number='9000000002'").fetchone()[0]
    assert stored != "secret1" and len(stored) > 30


# ---- SQL guard for the assistant -------------------------------------------
@pytest.mark.parametrize("sql", [
    "SELECT * FROM products",
    "SELECT name, price FROM products WHERE LOWER(name) LIKE '%pen%'",
    "WITH x AS (SELECT 1) SELECT * FROM x",
])
def test_guard_accepts_read_queries(sql):
    assert assistant._is_safe_read_query(sql)


@pytest.mark.parametrize("sql", [
    "DROP TABLE products",
    "SELECT 1; DELETE FROM products",
    "WITH x AS (SELECT 1) UPDATE products SET price = 0",
    "INSERT INTO products(name) VALUES ('x')",
    "",
])
def test_guard_rejects_unsafe_queries(sql):
    assert not assistant._is_safe_read_query(sql)


def test_database_itself_refuses_writes():
    """Defence in depth: even if the guard were bypassed, the connection is read-only."""
    with pytest.raises(Exception):
        db.execute_readonly_query(
            "INSERT INTO products(name,category,price,shelf_no,row_no) VALUES('a','b',1,1,1)")


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN LIMITATION: the assistant calls _is_safe_read_query() without products_only=True, "
    "so it will accept a query on the customers table (which holds password hashes)."))
def test_guard_should_restrict_tables_but_does_not():
    assert not assistant._is_safe_read_query("SELECT name, password_hash FROM customers")
