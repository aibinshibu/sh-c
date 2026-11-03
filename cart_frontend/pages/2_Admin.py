"""Password-gated admin tools for products, offers, customers, and sales."""

from __future__ import annotations

from datetime import date
import hmac
import os

import streamlit as st
from dotenv import load_dotenv

import api_client as db


load_dotenv()
st.set_page_config(page_title="Shop admin", page_icon="🔐", layout="wide")


def money(value: float) -> str:
    """Format a currency value consistently for this admin interface."""
    return f"Rs. {float(value):,.2f}"


def offer_status(offer: dict) -> str:
    """Calculate the display status for an offer based on today's date."""
    today = date.today().isoformat()
    if not offer["is_active"]:
        return "Inactive"
    if offer["valid_from"] > today:
        return "Upcoming"
    if offer["valid_until"] < today:
        return "Expired"
    return "Active"


def product_options(products: list[dict]) -> tuple[list[str], dict[str, int | None]]:
    """Return safe product selector labels and their corresponding product IDs."""
    labels = ["No specific product"]
    mapping: dict[str, int | None] = {"No specific product": None}
    for product in products:
        label = f"#{product['id']} - {product['name']}"
        labels.append(label)
        mapping[label] = product["id"]
    return labels, mapping


admin_password = os.getenv("ADMIN_PASSWORD")
if not admin_password:
    st.error("ADMIN_PASSWORD is not configured. Add it to .env before using the admin page.")
    st.stop()

if not st.session_state.get("is_admin"):
    st.title("Admin access")
    st.caption("This area is protected by the single shop-admin password.")
    with st.form("admin_password_form"):
        password = st.text_input("Admin password", type="password")
        submitted = st.form_submit_button("Unlock admin", type="primary")
    if submitted:
        if hmac.compare_digest(password, admin_password):
            st.session_state.is_admin = True
            st.rerun()
        else:
            st.error("Incorrect admin password.")
    if st.button("Back to shopping", use_container_width=True):
        st.switch_page("app.py")
    st.stop()

st.title("Shop administration")
top_left, top_right = st.columns((4, 1))
with top_left:
    st.caption("Manage shop data and review sales. Customer passwords are never shown here.")
with top_right:
    if st.button("Admin logout", use_container_width=True):
        st.session_state.pop("is_admin", None)
        st.rerun()

products_tab, offers_tab, customers_tab, sales_tab = st.tabs(
    ["Products", "Offers", "Customers", "Sales"]
)

with products_tab:
    search = st.text_input("Search products", placeholder="Name, category, or alternate name")
    threshold = int(st.number_input("Low-stock threshold", min_value=0, value=10, step=1))
    show_low_stock = st.checkbox(f"Show only items below {threshold}")
    product_rows = db.get_all_products(search, threshold if show_low_stock else None)
    st.dataframe(
        [
            {
                "ID": product["id"],
                "Name": product["name"],
                "Category": product["category"],
                "Price": money(product["price"]),
                "Stock": product["stock_quantity"],
                "Unit": product["unit"],
                "Shelf": product["shelf_no"],
                "Row": product["row_no"],
                "Alternate names": product["alternate_names"] or "",
            }
            for product in product_rows
        ],
        hide_index=True,
        use_container_width=True,
    )

    add_column, edit_column = st.columns(2)
    with add_column:
        st.subheader("Add product")
        with st.form("add_product_form", clear_on_submit=True):
            name = st.text_input("Name")
            category = st.text_input("Category")
            price = st.number_input("Price", min_value=0.0, step=1.0)
            stock = st.number_input("Stock quantity", min_value=0, step=1)
            unit = st.text_input("Unit", value="piece")
            shelf = st.number_input("Shelf number", min_value=0, step=1)
            row = st.number_input("Row number", min_value=0, step=1)
            aliases = st.text_input("Alternate names", help="Optional comma-separated names")
            submitted = st.form_submit_button("Add product", type="primary")
        if submitted:
            try:
                product_id = db.add_product(name, category, price, shelf, row, stock, unit, aliases)
                st.success(f"Product #{product_id} added.")
            except ValueError as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"Could not add product: {error}")

    with edit_column:
        st.subheader("Edit or delete product")
        all_products = db.get_all_products()
        labels, ids = product_options(all_products)
        product_label = st.selectbox("Choose product", labels[1:], key="edit_product_selector")
        product_id = ids[product_label]
        current = db.get_product_by_id(int(product_id))
        if current:
            with st.form(f"edit_product_form_{product_id}"):
                name = st.text_input("Name", value=current["name"])
                category = st.text_input("Category", value=current["category"])
                price = st.number_input("Price", min_value=0.0, value=float(current["price"]), step=1.0)
                stock = st.number_input("Stock quantity", min_value=0, value=int(current["stock_quantity"]), step=1)
                unit = st.text_input("Unit", value=current["unit"])
                shelf = st.number_input("Shelf number", min_value=0, value=int(current["shelf_no"]), step=1)
                row = st.number_input("Row number", min_value=0, value=int(current["row_no"]), step=1)
                aliases = st.text_input("Alternate names", value=current["alternate_names"] or "")
                submitted = st.form_submit_button("Save product")
            if submitted:
                try:
                    db.update_product(product_id, name, category, price, shelf, row, stock, unit, aliases)
                    st.success("Product updated.")
                except ValueError as error:
                    st.error(str(error))
                except Exception as error:
                    st.error(f"Could not update product: {error}")

            with st.form(f"delete_product_form_{product_id}"):
                confirm = st.checkbox(f"I want to delete {current['name']}")
                deleted = st.form_submit_button("Delete product")
            if deleted:
                if not confirm:
                    st.warning("Tick the confirmation box before deleting.")
                else:
                    try:
                        db.delete_product(product_id)
                        st.success("Product deleted.")
                    except ValueError as error:
                        st.error(str(error))
                    except Exception as error:
                        st.error(f"Could not delete product: {error}")

with offers_tab:
    all_products = db.get_all_products()
    selector_labels, selector_ids = product_options(all_products)
    offers = db.get_all_offers()
    st.dataframe(
        [
            {
                "ID": offer["id"],
                "Offer": offer["offer_text"],
                "Target": offer["target_category"] or offer["target_product_name"] or "Store-wide",
                "Discount": (
                    f"{offer['discount_value']:.0f}%"
                    if offer["discount_type"] == "percentage"
                    else money(offer["discount_value"])
                ),
                "Status": offer_status(offer),
                "From": offer["valid_from"],
                "Until": offer["valid_until"],
            }
            for offer in offers
        ],
        hide_index=True,
        use_container_width=True,
    )

    add_column, edit_column = st.columns(2)
    with add_column:
        st.subheader("Add offer")
        with st.form("add_offer_form", clear_on_submit=True):
            offer_text = st.text_input("Offer text")
            product_label = st.selectbox("Target product", selector_labels)
            target_category = st.text_input("Target category", help="Leave both target fields blank for store-wide.")
            discount_value = st.number_input("Discount value", min_value=0.0, step=1.0)
            discount_type = st.selectbox("Discount type", ["flat", "percentage"])
            valid_from = st.date_input("Valid from", value=date.today())
            valid_until = st.date_input("Valid until", value=date.today())
            active = st.checkbox("Enabled", value=True)
            submitted = st.form_submit_button("Add offer", type="primary")
        if submitted:
            try:
                offer_id = db.add_offer(
                    offer_text, selector_ids[product_label], target_category, discount_value,
                    discount_type, valid_from.isoformat(), valid_until.isoformat(), active,
                )
                st.success(f"Offer #{offer_id} added.")
            except ValueError as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"Could not add offer: {error}")

    with edit_column:
        st.subheader("Edit or delete offer")
        if offers:
            offer_label_map = {f"#{offer['id']} - {offer['offer_text']}": offer["id"] for offer in offers}
            selected_offer_label = st.selectbox("Choose offer", list(offer_label_map), key="edit_offer_selector")
            current = db.get_offer_by_id(offer_label_map[selected_offer_label])
            if current:
                current_product_label = next(
                    (label for label, product_id in selector_ids.items() if product_id == current["target_product_id"]),
                    "No specific product",
                )
                with st.form(f"edit_offer_form_{current['id']}"):
                    offer_text = st.text_input("Offer text", value=current["offer_text"])
                    product_label = st.selectbox(
                        "Target product", selector_labels, index=selector_labels.index(current_product_label)
                    )
                    target_category = st.text_input("Target category", value=current["target_category"] or "")
                    discount_value = st.number_input(
                        "Discount value", min_value=0.0, value=float(current["discount_value"]), step=1.0
                    )
                    discount_type = st.selectbox(
                        "Discount type", ["flat", "percentage"],
                        index=["flat", "percentage"].index(current["discount_type"]),
                    )
                    valid_from = st.date_input("Valid from", value=date.fromisoformat(current["valid_from"]))
                    valid_until = st.date_input("Valid until", value=date.fromisoformat(current["valid_until"]))
                    active = st.checkbox("Enabled", value=bool(current["is_active"]))
                    submitted = st.form_submit_button("Save offer")
                if submitted:
                    try:
                        db.update_offer(
                            current["id"], offer_text, selector_ids[product_label], target_category,
                            discount_value, discount_type, valid_from.isoformat(), valid_until.isoformat(), active,
                        )
                        st.success("Offer updated.")
                    except ValueError as error:
                        st.error(str(error))
                    except Exception as error:
                        st.error(f"Could not update offer: {error}")

                with st.form(f"delete_offer_form_{current['id']}"):
                    confirm = st.checkbox("I want to delete this offer")
                    deleted = st.form_submit_button("Delete offer")
                if deleted:
                    if not confirm:
                        st.warning("Tick the confirmation box before deleting.")
                    else:
                        try:
                            db.delete_offer(current["id"])
                            st.success("Offer deleted.")
                        except Exception as error:
                            st.error(f"Could not delete offer: {error}")
        else:
            st.caption("There are no offers to edit yet.")

with customers_tab:
    st.subheader("Customers")
    st.caption("Password hashes are deliberately never queried or displayed here.")
    st.dataframe(
        [
            {
                "Name": customer["name"],
                "Phone": customer["phone_number"],
                "Joined": customer["join_date"],
                "Orders": customer["order_count"],
                "Total spent": money(customer["total_spent"]),
            }
            for customer in db.get_customer_summaries()
        ],
        hide_index=True,
        use_container_width=True,
    )

with sales_tab:
    st.subheader("Sales")
    summary = db.get_sales_summary()
    today_column, week_column = st.columns(2)
    today_column.metric("Today's sales", money(summary["today_total"]))
    week_column.metric("This week's sales", money(summary["week_total"]))
    sales_orders = db.get_sales_orders()
    if not sales_orders:
        st.caption("No sales have been recorded yet.")
    for order in sales_orders:
        with st.expander(
            f"Order #{order['id']} - {order['order_date'][:10]} - {order['customer_name']} - {money(order['total_amount'])}"
        ):
            st.dataframe(
                [
                    {
                        "Product": item["product_name"],
                        "Qty": item["quantity"],
                        "Price at sale": money(item["price_at_sale"]),
                        "Subtotal": money(item["subtotal"]),
                    }
                    for item in order["items"]
                ],
                hide_index=True,
                use_container_width=True,
            )

if st.button("Back to shopping", key="admin_back_to_shop", use_container_width=True):
    st.switch_page("app.py")
