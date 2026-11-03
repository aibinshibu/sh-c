"""Test 5: two clients order the last unit at the same instant.

Exactly one order may succeed, the other must be rejected, and stock must end at
zero. Runs on a temporary database copy, so no test product is left behind.
"""
import threading

from fastapi.testclient import TestClient


def test_last_unit_cannot_be_oversold(client, headers):
    import db
    from main import app

    pid = client.post("/products", headers=headers, json={
        "name": "Race Product", "category": "tests", "price": 99.0, "shelf_no": 99,
        "row_no": 99, "stock_quantity": 1, "unit": "piece", "alternate_names": "race_item",
    }).json()
    customers = [c["id"] for c in db.get_customer_summaries()][:2]
    assert len(customers) == 2, "need two customers in the database"

    results, barrier = [], threading.Barrier(2)

    def order(customer_id):
        c = TestClient(app)
        barrier.wait()
        r = c.post("/orders", headers=headers,
                   json={"customer_id": customer_id, "items": [{"product_id": pid, "quantity": 1}]})
        results.append((r.status_code, r.json()))

    threads = [threading.Thread(target=order, args=(cid,)) for cid in customers]
    [t.start() for t in threads]
    [t.join() for t in threads]

    codes = sorted(code for code, _ in results)
    assert codes == [200, 400], f"expected one success and one rejection, got {codes}"
    rejected = next(body for code, body in results if code == 400)
    assert "stock" in rejected["detail"].lower()
    assert client.get(f"/products/{pid}", headers=headers).json()["stock_quantity"] == 0
