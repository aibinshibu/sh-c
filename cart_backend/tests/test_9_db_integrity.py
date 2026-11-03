"""Test 9: transactional order placement and database integrity rules."""
import pytest

import db


@pytest.fixture()
def customer():
    return db.get_customer_summaries()[0]["id"]


def stock_of(pid):
    return db.get_product_by_id(pid)["stock_quantity"]


def test_quote_total_is_computed_from_database_prices():
    pen = db.get_product_by_detector_label("pen")
    book = db.get_product_by_detector_label("book")
    total = db.quote_total([{"product_id": pen["id"], "quantity": 3},
                            {"product_id": book["id"], "quantity": 1}])
    assert total == pytest.approx(3 * 10.0 + 40.0)


def test_quote_rejects_empty_cart_and_overstock():
    with pytest.raises(ValueError, match="empty"):
        db.quote_total([])
    pen = db.get_product_by_detector_label("pen")
    with pytest.raises(ValueError, match="in stock"):
        db.quote_total([{"product_id": pen["id"], "quantity": 10**6}])


def test_place_order_reduces_stock_atomically(customer):
    pen = db.get_product_by_detector_label("pen")
    before = stock_of(pen["id"])
    order = db.place_order(customer, [{"product_id": pen["id"], "quantity": 2}])
    assert order["total_amount"] == pytest.approx(20.0)
    assert stock_of(pen["id"]) == before - 2


def test_failed_order_rolls_back_everything(customer):
    """One valid line plus one impossible line: nothing may be committed."""
    pen = db.get_product_by_detector_label("pen")
    before = stock_of(pen["id"])
    with pytest.raises(ValueError):
        db.place_order(customer, [{"product_id": pen["id"], "quantity": 1},
                                  {"product_id": pen["id"], "quantity": 10**6}])
    assert stock_of(pen["id"]) == before


def test_negative_quantity_is_rejected(customer):
    pen = db.get_product_by_detector_label("pen")
    with pytest.raises(ValueError, match="positive"):
        db.place_order(customer, [{"product_id": pen["id"], "quantity": -2}])


def test_payment_id_cannot_be_reused(customer):
    pen = db.get_product_by_detector_label("pen")
    items = [{"product_id": pen["id"], "quantity": 1}]
    db.place_order(customer, items, payment_id="pay_TEST1", payment_link_id="plink_1")
    with pytest.raises(ValueError, match="already been used"):
        db.place_order(customer, items, payment_id="pay_TEST1")


def test_product_with_sales_history_cannot_be_deleted(customer):
    pen = db.get_product_by_detector_label("pen")
    db.place_order(customer, [{"product_id": pen["id"], "quantity": 1}])
    with pytest.raises(ValueError, match="order history"):
        db.delete_product(pen["id"])


@pytest.mark.parametrize("kwargs,msg", [
    (dict(discount_value=150, discount_type="percentage",
          valid_from="2026-10-01", valid_until="2026-10-30"), "100"),
    (dict(discount_value=10, discount_type="flat",
          valid_from="2026-10-30", valid_until="2026-10-01"), "must not be before"),
])
def test_invalid_offers_are_rejected(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        db.add_offer("test offer", None, None, kwargs["discount_value"], kwargs["discount_type"],
                     kwargs["valid_from"], kwargs["valid_until"])
