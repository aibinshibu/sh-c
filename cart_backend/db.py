"""Shared, parameterized SQLite data access for the shopping-cart app."""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable

from werkzeug.security import check_password_hash, generate_password_hash


DB_PATH = Path(__file__).resolve().with_name("supermarket1.db")
PHONE_PATTERN = re.compile(r"^\d{8,15}$")
DISCOUNT_TYPES = {"flat", "percentage"}


def _connect(*, read_only: bool = False) -> sqlite3.Connection:
    """Open a configured SQLite connection for internal data-access functions."""
    if read_only:
        connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    else:
        connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _rows_as_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    """Convert SQLite rows to plain dictionaries for Streamlit rendering."""
    return [dict(row) for row in rows]


def _require_text(value: str, field_name: str) -> str:
    """Return non-empty trimmed text or raise ValueError for invalid form input."""
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} is required.")
    return cleaned


def _non_negative(value: float | int, field_name: str, *, integer: bool = False) -> float | int:
    """Validate a non-negative number and return its normalized numeric value."""
    try:
        normalized = int(value) if integer else float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a number.") from error
    if normalized < 0:
        raise ValueError(f"{field_name} cannot be negative.")
    return normalized


def validate_phone_number(phone_number: str) -> str:
    """Validate and return a digits-only customer phone number."""
    phone = str(phone_number).strip()
    if not PHONE_PATTERN.fullmatch(phone):
        raise ValueError("Phone number must contain 8 to 15 digits only.")
    return phone


def _validate_date(value: str, field_name: str) -> str:
    """Validate an ISO date string and return it in ISO format."""
    try:
        return date.fromisoformat(str(value)).isoformat()
    except ValueError as error:
        raise ValueError(f"{field_name} must be a valid date.") from error


def get_schema_sql() -> str:
    """Return SQL definitions for all application tables as a single string."""
    connection = _connect(read_only=True)
    try:
        rows = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = ? AND name NOT LIKE ? ORDER BY name",
            ("table", "sqlite_%"),
        ).fetchall()
        return "\n".join(row["sql"] for row in rows if row["sql"])
    finally:
        connection.close()


def execute_readonly_query(sql: str) -> list[tuple[Any, ...]]:
    """Execute a read-only generated SQL query and return its result tuples.

    SQLite's authorizer denies write, schema, and attachment actions even if a
    caller supplies malicious SQL. The caller is responsible for further query
    shape validation appropriate to its use case.
    """
    connection = _connect(read_only=True)
    try:
        allowed_actions = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}

        def authorizer(action: int, _arg1: str | None, _arg2: str | None, _db: str | None, _source: str | None) -> int:
            return sqlite3.SQLITE_OK if action in allowed_actions else sqlite3.SQLITE_DENY

        connection.set_authorizer(authorizer)
        return [tuple(row) for row in connection.execute(sql).fetchall()]
    finally:
        connection.close()


def get_all_products(search: str = "", low_stock_threshold: int | None = None) -> list[dict[str, Any]]:
    """Return products matching optional name/category/alias search and stock limit."""
    sql = "SELECT * FROM products WHERE 1 = 1"
    parameters: list[Any] = []
    if search.strip():
        search_value = f"%{search.strip().lower()}%"
        sql += " AND (LOWER(name) LIKE ? OR LOWER(category) LIKE ? OR LOWER(COALESCE(alternate_names, '')) LIKE ?)"
        parameters.extend([search_value, search_value, search_value])
    if low_stock_threshold is not None:
        sql += " AND stock_quantity < ?"
        parameters.append(_non_negative(low_stock_threshold, "Low-stock threshold", integer=True))
    sql += " ORDER BY name COLLATE NOCASE"

    connection = _connect(read_only=True)
    try:
        return _rows_as_dicts(connection.execute(sql, parameters).fetchall())
    finally:
        connection.close()


def get_product_by_id(product_id: int) -> dict[str, Any] | None:
    """Return one product dictionary by ID, or None when it does not exist."""
    connection = _connect(read_only=True)
    try:
        row = connection.execute("SELECT * FROM products WHERE id = ?", (int(product_id),)).fetchone()
        return dict(row) if row else None
    finally:
        connection.close()


def get_product_by_name(name: str) -> dict[str, Any] | None:
    """Return one product by case-insensitive exact product name, or None."""
    connection = _connect(read_only=True)
    try:
        row = connection.execute(
            "SELECT * FROM products WHERE LOWER(name) = LOWER(?) LIMIT 1", (str(name).strip(),)
        ).fetchone()
        return dict(row) if row else None
    finally:
        connection.close()


def get_all_product_prices() -> list[tuple[str, float]]:
    """Return every product's canonical name and current price for detector matching."""
    connection = _connect(read_only=True)
    try:
        return [(row["name"], float(row["price"])) for row in connection.execute("SELECT name, price FROM products")]
    finally:
        connection.close()


def _normalise_label(value: str) -> str:
    """Lowercase and strip all non-alphanumeric characters for label matching."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def get_product_by_detector_label(detector_label: str) -> dict[str, Any] | None:
    """Return the unique product matching a detector class label, or None.

    Matching rules (applied in order, short-circuit on first hit):
    1. Normalised product ``name`` equals normalised label.
    2. Any comma-separated token in ``alternate_names`` equals normalised label.

    If more than one product matches the same normalised label a warning is
    printed and ``None`` is returned so the caller can skip the detection
    rather than silently billing the wrong product.
    """
    normalised = _normalise_label(detector_label)
    if not normalised:
        return None

    connection = _connect(read_only=True)
    try:
        rows = connection.execute("SELECT * FROM products").fetchall()
    finally:
        connection.close()

    matches: list[dict[str, Any]] = []
    for row in rows:
        product = dict(row)
        if _normalise_label(product["name"]) == normalised:
            matches.append(product)
            continue
        raw_alts: str = product.get("alternate_names") or ""
        for token in raw_alts.split(","):
            if _normalise_label(token) == normalised:
                matches.append(product)
                break

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(m["name"] for m in matches)
        print(
            f"[detector] WARNING: label {detector_label!r} matched multiple products "
            f"({names}); skipping to avoid wrong billing."
        )
    return None


def add_product(
    name: str,
    category: str,
    price: float,
    shelf_no: int,
    row_no: int,
    stock_quantity: int,
    unit: str = "piece",
    alternate_names: str | None = None,
) -> int:
    """Create a validated product and return its new integer ID."""
    values = (
        _require_text(name, "Product name"),
        _require_text(category, "Category"),
        _non_negative(price, "Price"),
        _non_negative(shelf_no, "Shelf number", integer=True),
        _non_negative(row_no, "Row number", integer=True),
        _non_negative(stock_quantity, "Stock quantity", integer=True),
        _require_text(unit, "Unit"),
        str(alternate_names).strip() or None if alternate_names is not None else None,
    )
    connection = _connect()
    try:
        cursor = connection.execute(
            """
            INSERT INTO products
            (name, category, price, shelf_no, row_no, stock_quantity, unit, alternate_names)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        connection.commit()
        return int(cursor.lastrowid)
    finally:
        connection.close()


def update_product(
    product_id: int,
    name: str,
    category: str,
    price: float,
    shelf_no: int,
    row_no: int,
    stock_quantity: int,
    unit: str,
    alternate_names: str | None,
) -> bool:
    """Update one validated product and return whether a row was changed."""
    values = (
        _require_text(name, "Product name"),
        _require_text(category, "Category"),
        _non_negative(price, "Price"),
        _non_negative(shelf_no, "Shelf number", integer=True),
        _non_negative(row_no, "Row number", integer=True),
        _non_negative(stock_quantity, "Stock quantity", integer=True),
        _require_text(unit, "Unit"),
        str(alternate_names).strip() or None if alternate_names is not None else None,
        int(product_id),
    )
    connection = _connect()
    try:
        cursor = connection.execute(
            """
            UPDATE products
            SET name = ?, category = ?, price = ?, shelf_no = ?, row_no = ?,
                stock_quantity = ?, unit = ?, alternate_names = ?
            WHERE id = ?
            """,
            values,
        )
        connection.commit()
        return cursor.rowcount == 1
    finally:
        connection.close()


def update_stock(product_id: int, stock_quantity: int) -> bool:
    """Set a product's non-negative stock quantity and return success status."""
    connection = _connect()
    try:
        cursor = connection.execute(
            "UPDATE products SET stock_quantity = ? WHERE id = ?",
            (_non_negative(stock_quantity, "Stock quantity", integer=True), int(product_id)),
        )
        connection.commit()
        return cursor.rowcount == 1
    finally:
        connection.close()


def delete_product(product_id: int) -> bool:
    """Delete an unused product and return success; raise ValueError if it has sales history."""
    connection = _connect()
    try:
        try:
            cursor = connection.execute("DELETE FROM products WHERE id = ?", (int(product_id),))
            connection.commit()
            return cursor.rowcount == 1
        except sqlite3.IntegrityError as error:
            connection.rollback()
            raise ValueError("Products with order history cannot be deleted.") from error
    finally:
        connection.close()


def get_customer_by_phone(phone_number: str) -> dict[str, Any] | None:
    """Return a safe customer profile by phone number without its password hash."""
    phone = validate_phone_number(phone_number)
    connection = _connect(read_only=True)
    try:
        row = connection.execute(
            "SELECT id, name, phone_number, join_date FROM customers WHERE phone_number = ?", (phone,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        connection.close()


def create_customer(name: str, phone_number: str, password: str) -> int:
    """Validate, hash, and create a customer account; return its new customer ID."""
    name = _require_text(name, "Name")
    phone = validate_phone_number(phone_number)
    if len(password) < 6:
        raise ValueError("Password must have at least 6 characters.")
    connection = _connect()
    try:
        try:
            cursor = connection.execute(
                "INSERT INTO customers (name, phone_number, password_hash, join_date) VALUES (?, ?, ?, ?)",
                (name, phone, generate_password_hash(password), date.today().isoformat()),
            )
            connection.commit()
            return int(cursor.lastrowid)
        except sqlite3.IntegrityError as error:
            connection.rollback()
            raise ValueError("An account already exists for this phone number.") from error
    finally:
        connection.close()


def verify_login(phone_number: str, password: str) -> dict[str, Any] | None:
    """Verify customer credentials and return a safe customer profile, or None."""
    phone = validate_phone_number(phone_number)
    connection = _connect(read_only=True)
    try:
        row = connection.execute(
            "SELECT id, name, phone_number, join_date, password_hash FROM customers WHERE phone_number = ?",
            (phone,),
        ).fetchone()
        if row is None or not check_password_hash(row["password_hash"], password):
            return None
        return {key: row[key] for key in ("id", "name", "phone_number", "join_date")}
    finally:
        connection.close()


def get_active_offers(customer_id: int) -> list[dict[str, Any]]:
    """Return today's active store-wide and customer-relevant category/product offers."""
    today = date.today().isoformat()
    connection = _connect(read_only=True)
    try:
        rows = connection.execute(
            """
            SELECT DISTINCT o.*, p.name AS target_product_name
            FROM offers AS o
            LEFT JOIN products AS p ON p.id = o.target_product_id
            WHERE o.is_active = 1
              AND o.valid_from <= ? AND o.valid_until >= ?
              AND (
                    (o.target_category IS NULL AND o.target_product_id IS NULL)
                    OR o.target_category IN (
                        SELECT DISTINCT pr.category
                        FROM orders AS ord
                        JOIN order_items AS oi ON oi.order_id = ord.id
                        JOIN products AS pr ON pr.id = oi.product_id
                        WHERE ord.customer_id = ?
                    )
                    OR o.target_product_id IN (
                        SELECT DISTINCT oi.product_id
                        FROM orders AS ord
                        JOIN order_items AS oi ON oi.order_id = ord.id
                        WHERE ord.customer_id = ?
                    )
                  )
            ORDER BY o.valid_until, o.id
            """,
            (today, today, int(customer_id), int(customer_id)),
        ).fetchall()
        return _rows_as_dicts(rows)
    finally:
        connection.close()


def get_all_offers() -> list[dict[str, Any]]:
    """Return every offer with optional target product names for admin display."""
    connection = _connect(read_only=True)
    try:
        rows = connection.execute(
            """
            SELECT o.*, p.name AS target_product_name
            FROM offers AS o LEFT JOIN products AS p ON p.id = o.target_product_id
            ORDER BY o.valid_until DESC, o.id DESC
            """
        ).fetchall()
        return _rows_as_dicts(rows)
    finally:
        connection.close()


def get_offer_by_id(offer_id: int) -> dict[str, Any] | None:
    """Return one offer dictionary by ID, or None if the ID is not found."""
    connection = _connect(read_only=True)
    try:
        row = connection.execute("SELECT * FROM offers WHERE id = ?", (int(offer_id),)).fetchone()
        return dict(row) if row else None
    finally:
        connection.close()


def _offer_values(
    offer_text: str,
    target_product_id: int | None,
    target_category: str | None,
    discount_value: float,
    discount_type: str,
    valid_from: str,
    valid_until: str,
    is_active: bool | int,
) -> tuple[Any, ...]:
    """Validate and normalize offer inputs for create and update operations."""
    start = _validate_date(valid_from, "Valid-from date")
    end = _validate_date(valid_until, "Valid-until date")
    if start > end:
        raise ValueError("Valid-until date must not be before the valid-from date.")
    discount_type = str(discount_type).lower()
    if discount_type not in DISCOUNT_TYPES:
        raise ValueError("Discount type must be flat or percentage.")
    if discount_type == "percentage" and float(discount_value) > 100:
        raise ValueError("Percentage discount cannot exceed 100.")
    return (
        _require_text(offer_text, "Offer text"),
        int(target_product_id) if target_product_id else None,
        str(target_category).strip() or None if target_category else None,
        _non_negative(discount_value, "Discount value"),
        discount_type,
        start,
        end,
        1 if bool(is_active) else 0,
    )


def add_offer(
    offer_text: str,
    target_product_id: int | None,
    target_category: str | None,
    discount_value: float,
    discount_type: str,
    valid_from: str,
    valid_until: str,
    is_active: bool | int = True,
) -> int:
    """Create a validated offer and return its new integer ID."""
    connection = _connect()
    try:
        cursor = connection.execute(
            """
            INSERT INTO offers
            (offer_text, target_product_id, target_category, discount_value, discount_type,
             valid_from, valid_until, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _offer_values(
                offer_text, target_product_id, target_category, discount_value,
                discount_type, valid_from, valid_until, is_active,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)
    finally:
        connection.close()


def update_offer(
    offer_id: int,
    offer_text: str,
    target_product_id: int | None,
    target_category: str | None,
    discount_value: float,
    discount_type: str,
    valid_from: str,
    valid_until: str,
    is_active: bool | int,
) -> bool:
    """Update a validated offer and return whether exactly one row was changed."""
    connection = _connect()
    try:
        values = _offer_values(
            offer_text, target_product_id, target_category, discount_value,
            discount_type, valid_from, valid_until, is_active,
        ) + (int(offer_id),)
        cursor = connection.execute(
            """
            UPDATE offers
            SET offer_text = ?, target_product_id = ?, target_category = ?, discount_value = ?,
                discount_type = ?, valid_from = ?, valid_until = ?, is_active = ?
            WHERE id = ?
            """,
            values,
        )
        connection.commit()
        return cursor.rowcount == 1
    finally:
        connection.close()


def delete_offer(offer_id: int) -> bool:
    """Delete one offer and return whether it existed."""
    connection = _connect()
    try:
        cursor = connection.execute("DELETE FROM offers WHERE id = ?", (int(offer_id),))
        connection.commit()
        return cursor.rowcount == 1
    finally:
        connection.close()


def get_recommendations(customer_id: int, limit: int = 8) -> list[dict[str, Any]]:
    """Return products from a customer's most-purchased categories, or an empty list."""
    connection = _connect(read_only=True)
    try:
        rows = connection.execute(
            """
            WITH category_rank AS (
                SELECT p.category, SUM(oi.quantity) AS quantity_bought
                FROM orders AS o
                JOIN order_items AS oi ON oi.order_id = o.id
                JOIN products AS p ON p.id = oi.product_id
                WHERE o.customer_id = ?
                GROUP BY p.category
                ORDER BY quantity_bought DESC, p.category
                LIMIT 2
            )
            SELECT p.*
            FROM products AS p
            JOIN category_rank AS cr ON cr.category = p.category
            WHERE p.stock_quantity > 0
            ORDER BY cr.quantity_bought DESC, p.name COLLATE NOCASE
            LIMIT ?
            """,
            (int(customer_id), _non_negative(limit, "Recommendation limit", integer=True)),
        ).fetchall()
        return _rows_as_dicts(rows)
    finally:
        connection.close()


def get_customer_orders(customer_id: int) -> list[dict[str, Any]]:
    """Return a customer's orders with embedded line items, newest order first."""
    connection = _connect(read_only=True)
    try:
        orders = _rows_as_dicts(
            connection.execute(
                "SELECT id, order_date, total_amount FROM orders WHERE customer_id = ? ORDER BY order_date DESC, id DESC",
                (int(customer_id),),
            ).fetchall()
        )
        for order in orders:
            items = connection.execute(
                """
                SELECT oi.product_id, p.name AS product_name, oi.quantity, oi.price_at_sale,
                       ROUND(oi.quantity * oi.price_at_sale, 2) AS subtotal
                FROM order_items AS oi JOIN products AS p ON p.id = oi.product_id
                WHERE oi.order_id = ? ORDER BY p.name COLLATE NOCASE
                """,
                (order["id"],),
            ).fetchall()
            order["items"] = _rows_as_dicts(items)
        return orders
    finally:
        connection.close()


def _ensure_payment_columns(connection: sqlite3.Connection) -> None:
    """Add Razorpay columns to an existing orders table (safe to call repeatedly)."""
    existing = {row["name"] for row in connection.execute("PRAGMA table_info(orders)")}
    for column in ("payment_id", "payment_link_id", "payment_status"):
        if column not in existing:
            connection.execute(f"ALTER TABLE orders ADD COLUMN {column} TEXT")
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_orders_payment_id ON orders(payment_id) WHERE payment_id IS NOT NULL"
    )


def quote_total(items: Iterable[dict[str, Any]]) -> float:
    """Price a cart from the database (never trust a client-side total) and check stock."""
    quantities: Counter[int] = Counter()
    for item in items:
        quantities[int(item["product_id"])] += int(item["quantity"])
    if not quantities:
        raise ValueError("Cannot pay for an empty cart.")
    total = 0.0
    for product_id, quantity in quantities.items():
        product = get_product_by_id(product_id)
        if product is None:
            raise ValueError(f"Product ID {product_id} no longer exists.")
        if product["stock_quantity"] < quantity:
            raise ValueError(f"Only {product['stock_quantity']} {product['name']} item(s) are in stock.")
        total += quantity * float(product["price"])
    return round(total, 2)


def place_order(
    customer_id: int,
    items: Iterable[dict[str, Any]],
    payment_id: str | None = None,
    payment_link_id: str | None = None,
) -> dict[str, Any]:
    """Create an order and reduce stock in one transaction; return order details.

    When ``payment_id`` is given the order is stored as paid via Razorpay; a payment
    ID can only ever be attached to one order, so reruns cannot create duplicates.

    ``items`` must contain ``product_id`` and positive ``quantity`` values.
    The transaction rolls back fully if a customer/product is invalid or stock
    is insufficient, so orders and inventory never diverge.
    """
    quantities: Counter[int] = Counter()
    for item in items:
        try:
            product_id = int(item["product_id"])
            quantity = int(item["quantity"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Each cart item needs a product ID and whole-number quantity.") from error
        if quantity <= 0:
            raise ValueError("Order quantities must be positive.")
        quantities[product_id] += quantity
    if not quantities:
        raise ValueError("Cannot place an empty order.")

    connection = _connect()
    try:
        _ensure_payment_columns(connection)
        with connection:
            if payment_id and connection.execute(
                "SELECT 1 FROM orders WHERE payment_id = ?", (payment_id,)
            ).fetchone():
                raise ValueError("This payment has already been used for an order.")
            customer = connection.execute("SELECT id FROM customers WHERE id = ?", (int(customer_id),)).fetchone()
            if customer is None:
                raise ValueError("Please log in before placing an order.")

            line_items: list[dict[str, Any]] = []
            for product_id, quantity in quantities.items():
                product = connection.execute(
                    "SELECT id, name, price, stock_quantity FROM products WHERE id = ?", (product_id,)
                ).fetchone()
                if product is None:
                    raise ValueError(f"Product ID {product_id} no longer exists.")
                if product["stock_quantity"] < quantity:
                    raise ValueError(f"Only {product['stock_quantity']} {product['name']} item(s) are in stock.")
                line_items.append(
                    {
                        "product_id": product_id,
                        "product_name": product["name"],
                        "quantity": quantity,
                        "price_at_sale": float(product["price"]),
                    }
                )

            total = round(sum(item["quantity"] * item["price_at_sale"] for item in line_items), 2)
            order_date = datetime.now().astimezone().isoformat(timespec="seconds")
            cursor = connection.execute(
                """
                INSERT INTO orders (customer_id, order_date, total_amount, payment_id, payment_link_id, payment_status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (int(customer_id), order_date, total, payment_id, payment_link_id, "paid" if payment_id else None),
            )
            order_id = int(cursor.lastrowid)
            for item in line_items:
                connection.execute(
                    "INSERT INTO order_items (order_id, product_id, quantity, price_at_sale) VALUES (?, ?, ?, ?)",
                    (order_id, item["product_id"], item["quantity"], item["price_at_sale"]),
                )
                updated = connection.execute(
                    """
                    UPDATE products SET stock_quantity = stock_quantity - ?
                    WHERE id = ? AND stock_quantity >= ?
                    """,
                    (item["quantity"], item["product_id"], item["quantity"]),
                )
                if updated.rowcount != 1:
                    raise ValueError(f"Stock changed while placing the order for {item['product_name']}.")
        return {"id": order_id, "order_date": order_date, "total_amount": total, "items": line_items}
    finally:
        connection.close()


def get_customer_summaries() -> list[dict[str, Any]]:
    """Return admin-safe customer rows with order count and total spend, no hashes."""
    connection = _connect(read_only=True)
    try:
        rows = connection.execute(
            """
            SELECT c.id, c.name, c.phone_number, c.join_date,
                   COUNT(o.id) AS order_count, COALESCE(ROUND(SUM(o.total_amount), 2), 0) AS total_spent
            FROM customers AS c LEFT JOIN orders AS o ON o.customer_id = c.id
            GROUP BY c.id
            ORDER BY c.join_date DESC, c.name COLLATE NOCASE
            """
        ).fetchall()
        return _rows_as_dicts(rows)
    finally:
        connection.close()


def get_sales_summary() -> dict[str, float]:
    """Return total sales for today and from Monday through today."""
    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    connection = _connect(read_only=True)
    try:
        row = connection.execute(
            """
            SELECT
              COALESCE(SUM(CASE WHEN substr(order_date, 1, 10) = ? THEN total_amount END), 0) AS today_total,
              COALESCE(SUM(CASE WHEN substr(order_date, 1, 10) BETWEEN ? AND ? THEN total_amount END), 0) AS week_total
            FROM orders
            """,
            (today.isoformat(), week_start.isoformat(), today.isoformat()),
        ).fetchone()
        return {"today_total": float(row["today_total"]), "week_total": float(row["week_total"])}
    finally:
        connection.close()


def get_sales_orders() -> list[dict[str, Any]]:
    """Return every sale with customer name and line items for the admin sales tab."""
    connection = _connect(read_only=True)
    try:
        orders = _rows_as_dicts(
            connection.execute(
                """
                SELECT o.id, o.order_date, o.total_amount, c.name AS customer_name
                FROM orders AS o JOIN customers AS c ON c.id = o.customer_id
                ORDER BY o.order_date DESC, o.id DESC
                """
            ).fetchall()
        )
        for order in orders:
            order["items"] = _rows_as_dicts(
                connection.execute(
                    """
                    SELECT p.name AS product_name, oi.quantity, oi.price_at_sale,
                           ROUND(oi.quantity * oi.price_at_sale, 2) AS subtotal
                    FROM order_items AS oi JOIN products AS p ON p.id = oi.product_id
                    WHERE oi.order_id = ? ORDER BY p.name COLLATE NOCASE
                    """,
                    (order["id"],),
                ).fetchall()
            )
        return orders
    finally:
        connection.close()
