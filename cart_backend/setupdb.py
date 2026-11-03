"""Create a clean, realistic SQLite database for the shopping-cart app.

Running this script recreates ``supermarket1.db`` and therefore resets its
sample data. Use it for development/demo setup, not against live shop data.
"""

from datetime import date, timedelta
import sqlite3
from pathlib import Path

from werkzeug.security import generate_password_hash


DB_PATH = Path(__file__).resolve().with_name("supermarket1.db")


def setup_database(db_path: Path = DB_PATH) -> None:
    """Rebuild the demo database with schema, constraints, and seed records."""
    database_path = Path(db_path)
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    cursor = connection.cursor()

    cursor.executescript(
        """
        DROP TABLE IF EXISTS order_items;
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS offers;
        DROP TABLE IF EXISTS customers;
        DROP TABLE IF EXISTS products;

        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL CHECK (price >= 0),
            shelf_no INTEGER NOT NULL CHECK (shelf_no >= 0),
            row_no INTEGER NOT NULL CHECK (row_no >= 0),
            stock_quantity INTEGER NOT NULL DEFAULT 0 CHECK (stock_quantity >= 0),
            unit TEXT NOT NULL DEFAULT 'piece',
            alternate_names TEXT
        );

        CREATE TABLE customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone_number TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            join_date TEXT NOT NULL
        );

        CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            order_date TEXT NOT NULL,
            total_amount REAL NOT NULL CHECK (total_amount >= 0),
            payment_id TEXT,
            payment_link_id TEXT,
            payment_status TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );

        CREATE TABLE order_items (
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL CHECK (quantity > 0),
            price_at_sale REAL NOT NULL CHECK (price_at_sale >= 0),
            PRIMARY KEY (order_id, product_id),
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );

        CREATE TABLE offers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            offer_text TEXT NOT NULL,
            target_product_id INTEGER,
            target_category TEXT,
            discount_value REAL NOT NULL CHECK (discount_value >= 0),
            discount_type TEXT NOT NULL DEFAULT 'flat'
                CHECK (discount_type IN ('flat', 'percentage')),
            valid_from TEXT NOT NULL,
            valid_until TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
            FOREIGN KEY (target_product_id) REFERENCES products(id)
        );

        CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date DESC);
        CREATE INDEX idx_order_items_product ON order_items(product_id);
        CREATE INDEX idx_offers_dates ON offers(is_active, valid_from, valid_until);
        """
    )

    products = [
        ("pen", "stationery", 10.00, 1, 1, 120, "piece", "ball pen,ink pen"),
        ("calculator", "stationery", 250.00, 1, 2, 24, "piece", "scientific calculator"),
        ("chessboard", "games", 450.00, 2, 1, 18, "piece", "chess board,chess_board"),
        ("gum", "confectionery", 5.00, 3, 1, 220, "piece", "chewing gum"),
        ("happydent", "mouth freshener", 85.00, 3, 2, 55, "piece", "hyperdent,toothpaste"),
        ("pencil", "stationery", 5.00, 1, 3, 200, "piece", "lead pencil"),
        ("eraser", "stationery", 5.00, 1, 4, 130, "piece", "rubber"),
        ("notebook", "stationery", 50.00, 1, 5, 65, "piece", "exercise book"),
        ("marker", "stationery", 30.00, 1, 6, 40, "piece", "permanent marker"),
        ("ruler", "stationery", 20.00, 1, 7, 35, "piece", "scale"),
        ("shampoo", "personal care", 180.00, 4, 1, 42, "bottle", "hair wash"),
        ("soap", "personal care", 40.00, 4, 2, 90, "bar", "bath soap"),
        ("toothbrush", "personal care", 35.00, 4, 3, 48, "piece", "brush"),
        ("hand wash", "personal care", 120.00, 4, 4, 30, "bottle", "liquid soap"),
        ("biscuits", "food", 30.00, 5, 1, 80, "packet", "cookies"),
        ("chocolate", "confectionery", 50.00, 3, 3, 75, "bar", "candy"),
        ("potato chips", "snacks", 40.00, 3, 4, 70, "packet", "chips"),
        ("juice", "beverages", 60.00, 7, 1, 36, "bottle", "fruit juice"),
        ("milk", "dairy", 55.00, 5, 2, 40, "packet", "paal"),
        ("bread", "bakery", 45.00, 5, 3, 25, "loaf", "double roti"),
        ("rice", "groceries", 70.00, 6, 1, 100, "kg", "ari"),
        ("wheat flour", "groceries", 55.00, 6, 2, 85, "kg", "atta"),
        ("sugar", "groceries", 48.00, 6, 3, 75, "kg", "panchasara"),
        ("salt", "groceries", 25.00, 6, 4, 110, "kg", "uppu"),
        ("cooking oil", "groceries", 150.00, 6, 5, 50, "litre", "oil,enna"),
        ("water bottle", "beverages", 20.00, 7, 2, 96, "bottle", "drinking water"),
        ("coffee", "beverages", 180.00, 7, 3, 28, "packet", "coffee powder"),
        ("tea", "beverages", 140.00, 7, 4, 31, "packet", "tea powder"),
        ("football", "sports", 600.00, 2, 2, 9, "piece", "soccer ball"),
        ("cricket bat", "sports", 1200.00, 2, 3, 8, "piece", "bat"),
        ("tennis ball", "sports", 80.00, 2, 4, 20, "piece", "ball"),
        ("usb cable", "electronics", 150.00, 8, 1, 27, "piece", "charging cable"),
        ("earphones", "electronics", 500.00, 8, 2, 16, "piece", "headphones"),
        ("power bank", "electronics", 1200.00, 8, 3, 11, "piece", "portable charger"),
        ("chess_box", "games", 550.00, 2, 5, 12, "piece", "wooden chess box"), # Added chess_box (ID: 35)
        ("watch", "accessories", 2500.00, 9, 1, 15, "piece", "wrist watch,smartwatch"), # Added watch (ID: 36)
        ("book", "stationery", 40.00, 1, 8, 60, "piece", "textbook"),
        ("unomax pen", "stationery", 15.00, 1, 9, 90, "piece", "unomax_pen"),
    ]
    cursor.executemany(
        """
        INSERT INTO products
        (name, category, price, shelf_no, row_no, stock_quantity, unit, alternate_names)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        products,
    )

    today = date.today()
    customers = [
        ("Alice Sharma", "9876543210", generate_password_hash("alice123"), str(today - timedelta(days=365))),
        ("Rahul Verma", "9123456789", generate_password_hash("rahul123"), str(today - timedelta(days=300))),
        ("Aib", "9988776655", generate_password_hash("aib123"), str(today - timedelta(days=150))), # Added customer Aib (ID: 3)
    ]
    cursor.executemany(
        "INSERT INTO customers (name, phone_number, password_hash, join_date) VALUES (?, ?, ?, ?)",
        customers,
    )

    offers = [
        (
            "10% off all stationery",
            None,
            "stationery",
            10.00,
            "percentage",
            str(today - timedelta(days=7)),
            str(today + timedelta(days=30)),
            1,
        ),
        (
            "Save Rs. 20 on Coffee",
            27,
            None,
            20.00,
            "flat",
            str(today - timedelta(days=7)),
            str(today + timedelta(days=14)),
            1,
        ),
    ]
    cursor.executemany(
        """
        INSERT INTO offers
        (offer_text, target_product_id, target_category, discount_value, discount_type,
         valid_from, valid_until, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        offers,
    )

    # Order history for Alice (ID: 1)
    cursor.execute(
        "INSERT INTO orders (customer_id, order_date, total_amount) VALUES (?, ?, ?)",
        (1, str(today - timedelta(days=10)), 60.00),
    )
    order_id = cursor.lastrowid
    cursor.executemany(
        "INSERT INTO order_items (order_id, product_id, quantity, price_at_sale) VALUES (?, ?, ?, ?)",
        [(order_id, 8, 1, 50.00), (order_id, 1, 1, 10.00)],
    )

    # Purchase history for Aib (ID: 3) buying the newly added watch (ID: 36) and chess_box (ID: 35)
    cursor.execute(
        "INSERT INTO orders (customer_id, order_date, total_amount) VALUES (?, ?, ?)",
        (3, str(today - timedelta(days=5)), 3050.00),
    )
    aib_order_id = cursor.lastrowid
    cursor.executemany(
        "INSERT INTO order_items (order_id, product_id, quantity, price_at_sale) VALUES (?, ?, ?, ?)",
        [(aib_order_id, 35, 1, 550.00), (aib_order_id, 36, 1, 2500.00)],
    )

    connection.commit()
    connection.close()
    print(f"Database '{database_path.name}' created and seeded successfully.")


if __name__ == "__main__":
    setup_database()