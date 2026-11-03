"""Test 7: detector label -> catalogue product resolution (deterministic, no LLM)."""
import io
import sqlite3
from unittest.mock import patch

import pytest

import assistant
import db

UNKNOWN = "zzz_not_a_product"   # a label that can never exist in the catalogue


@pytest.mark.parametrize("label,name,price", [
    ("pen", "pen", 10.0),                         # exact name
    ("book", "book", 40.0),                       # exact name (added to catalogue later)
    ("gum", "gum", 5.0),
    ("chess_box", "chess_box", 550.0),            # exact name with underscore
    ("chess_board", "chessboard", 450.0),         # normalisation: chess_board == chessboard
    ("unomax_pen", "unomax pen", 15.0),           # normalisation: unomax_pen == unomax pen
])
def test_known_labels_resolve_to_one_product(label, name, price):
    product = db.get_product_by_detector_label(label)
    assert product is not None, f"{label!r} should resolve"
    assert product["name"] == name
    assert float(product["price"]) == pytest.approx(price)


def test_every_detector_class_has_exactly_one_product():
    """All nine classes in the model metadata must be billable."""
    import yaml
    from conftest import BACKEND_DIR
    meta = yaml.safe_load((BACKEND_DIR / "models" / "metadata.yaml").read_text())
    names = meta["names"]
    labels = names.values() if isinstance(names, dict) else names
    missing = [l for l in labels if db.get_product_by_detector_label(l) is None]
    assert not missing, f"detector classes without a catalogue product: {missing}"


def test_unknown_label_returns_none():
    assert db.get_product_by_detector_label(UNKNOWN) is None


def test_ambiguous_label_returns_none_and_warns(tmp_db):
    with sqlite3.connect(tmp_db) as conn:
        conn.execute(
            "INSERT INTO products (name, category, price, shelf_no, row_no, stock_quantity, unit, alternate_names)"
            " VALUES ('fountain pen', 'stationery', 50.0, 1, 1, 10, 'piece', 'pen')")
    captured = io.StringIO()
    with patch("sys.stdout", captured):
        assert db.get_product_by_detector_label("pen") is None
    assert "WARNING" in captured.getvalue()


@pytest.mark.parametrize("label", ["", "   ", "\t\n"])
def test_empty_label_returns_none(label):
    assert db.get_product_by_detector_label(label) is None


def test_assistant_price_lookup_never_calls_gemini():
    def boom(*a, **k):
        raise RuntimeError("Gemini must not be called during price lookup")

    with patch.object(assistant, "ChatGoogleGenerativeAI", boom):
        found, source = assistant.get_product_price_for_detector_label("gum")
        assert found["name"] == "gum" and float(found["price"]) == 5.0
        assert found["label"] == "gum" and source == "db_exact_lookup"

        none_found, src = assistant.get_product_price_for_detector_label(UNKNOWN)
        assert none_found is None and src == "db_exact_lookup"

        empty, src = assistant.get_product_price_for_detector_label("")
        assert empty is None and src == "N/A"
