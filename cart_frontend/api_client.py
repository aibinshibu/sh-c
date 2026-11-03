"""Frontend client for the shopping-cart backend.

Every function here has the SAME NAME and the SAME ARGUMENTS as the function in the old
db.py, so the pages only need `import api_client as db`. Each one sends a request to the
FastAPI backend and returns the answer in the same shape db.py used to return.

Error mapping (so the pages keep working unchanged):
    * backend replies 400  ->  ValueError with the backend's message (friendly text for users)
    * anything else wrong  ->  BackendError (a RuntimeError) with a readable message
"""

from __future__ import annotations

import os
from typing import Any, Iterable

import requests
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "")

TIMEOUT = 30          # seconds for normal requests
CHAT_TIMEOUT = 90     # Gemini can be slow
UPLOAD_TIMEOUT = 300  # video uploads


class BackendError(RuntimeError):
    """The backend could not be reached or answered with an error."""


def _detail(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail")
        if detail:
            return str(detail)
    except Exception:
        pass
    return response.text[:200] or f"HTTP {response.status_code}"


def request(method: str, path: str, *, params: dict | None = None, json: Any = None,
            files: Any = None, timeout: float = TIMEOUT, allow_404: bool = False) -> Any:
    """Send one request to the backend and return the decoded JSON answer."""
    try:
        response = requests.request(
            method, f"{BACKEND_URL}{path}", params=params, json=json, files=files,
            headers={"X-API-Key": API_KEY}, timeout=timeout,
        )
    except requests.ConnectionError as error:
        raise BackendError(
            f"The shop server is not reachable at {BACKEND_URL}. Check that the backend is running."
        ) from error
    except requests.Timeout as error:
        raise BackendError("The shop server took too long to answer. Please try again.") from error

    if response.status_code == 404 and allow_404:
        return None
    if response.status_code == 400:
        raise ValueError(_detail(response))
    if response.status_code == 401:
        raise BackendError("The backend rejected the API key. Check API_KEY in the frontend .env file.")
    if response.status_code >= 400:
        raise BackendError(_detail(response))
    return response.json()


# ----------------------------------------------------------------------------------------
# Products
# ----------------------------------------------------------------------------------------
def get_all_products(search: str = "", low_stock_threshold: int | None = None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"search": search}
    if low_stock_threshold is not None:
        params["low_stock_threshold"] = low_stock_threshold
    return request("GET", "/products", params=params)


def get_product_by_id(product_id: int) -> dict[str, Any] | None:
    return request("GET", f"/products/{int(product_id)}")


def add_product(
    name: str, category: str, price: float, shelf_no: int, row_no: int, stock_quantity: int,
    unit: str = "piece", alternate_names: str | None = None,
) -> int:
    return request("POST", "/products", json={
        "name": name, "category": category, "price": price, "shelf_no": shelf_no, "row_no": row_no,
        "stock_quantity": stock_quantity, "unit": unit, "alternate_names": alternate_names,
    })


def update_product(
    product_id: int, name: str, category: str, price: float, shelf_no: int, row_no: int,
    stock_quantity: int, unit: str, alternate_names: str | None,
) -> bool:
    return request("PUT", f"/products/{int(product_id)}", json={
        "name": name, "category": category, "price": price, "shelf_no": shelf_no, "row_no": row_no,
        "stock_quantity": stock_quantity, "unit": unit, "alternate_names": alternate_names,
    })


def delete_product(product_id: int) -> bool:
    return request("DELETE", f"/products/{int(product_id)}")


# ----------------------------------------------------------------------------------------
# Customers
# ----------------------------------------------------------------------------------------
def create_customer(name: str, phone_number: str, password: str) -> int:
    return request("POST", "/customers", json={"name": name, "phone_number": phone_number, "password": password})


def verify_login(phone_number: str, password: str) -> dict[str, Any] | None:
    return request("POST", "/login", json={"phone_number": phone_number, "password": password})


def get_active_offers(customer_id: int) -> list[dict[str, Any]]:
    return request("GET", f"/customers/{int(customer_id)}/offers")


def get_recommendations(customer_id: int, limit: int = 8) -> list[dict[str, Any]]:
    return request("GET", f"/customers/{int(customer_id)}/recommendations", params={"limit": limit})


def get_customer_orders(customer_id: int) -> list[dict[str, Any]]:
    return request("GET", f"/customers/{int(customer_id)}/orders")


# ----------------------------------------------------------------------------------------
# Offers
# ----------------------------------------------------------------------------------------
def get_all_offers() -> list[dict[str, Any]]:
    return request("GET", "/offers")


def get_offer_by_id(offer_id: int) -> dict[str, Any] | None:
    return request("GET", f"/offers/{int(offer_id)}")


def add_offer(
    offer_text: str, target_product_id: int | None, target_category: str | None, discount_value: float,
    discount_type: str, valid_from: str, valid_until: str, is_active: bool | int = True,
) -> int:
    return request("POST", "/offers", json={
        "offer_text": offer_text, "target_product_id": target_product_id, "target_category": target_category,
        "discount_value": discount_value, "discount_type": discount_type,
        "valid_from": valid_from, "valid_until": valid_until, "is_active": is_active,
    })


def update_offer(
    offer_id: int, offer_text: str, target_product_id: int | None, target_category: str | None,
    discount_value: float, discount_type: str, valid_from: str, valid_until: str, is_active: bool | int,
) -> bool:
    return request("PUT", f"/offers/{int(offer_id)}", json={
        "offer_text": offer_text, "target_product_id": target_product_id, "target_category": target_category,
        "discount_value": discount_value, "discount_type": discount_type,
        "valid_from": valid_from, "valid_until": valid_until, "is_active": is_active,
    })


def delete_offer(offer_id: int) -> bool:
    return request("DELETE", f"/offers/{int(offer_id)}")


# ----------------------------------------------------------------------------------------
# Orders
# ----------------------------------------------------------------------------------------
def quote_total(items: Iterable[dict[str, Any]]) -> float:
    return request("POST", "/orders/quote", json={"items": list(items)})


def place_order(
    customer_id: int, items: Iterable[dict[str, Any]],
    payment_id: str | None = None, payment_link_id: str | None = None,
) -> dict[str, Any]:
    return request("POST", "/orders", json={
        "customer_id": customer_id, "items": list(items),
        "payment_id": payment_id, "payment_link_id": payment_link_id,
    })


# ----------------------------------------------------------------------------------------
# Admin reports
# ----------------------------------------------------------------------------------------
def get_customer_summaries() -> list[dict[str, Any]]:
    return request("GET", "/admin/customers")


def get_sales_summary() -> dict[str, float]:
    return request("GET", "/admin/sales-summary")


def get_sales_orders() -> list[dict[str, Any]]:
    return request("GET", "/admin/orders")


# ----------------------------------------------------------------------------------------
# Gemini shopping assistant (was backend.get_shopping_assistant_response)
# ----------------------------------------------------------------------------------------
def get_shopping_assistant_response(user_query: str) -> tuple[str, str]:
    reply = request("POST", "/chat", json={"query": user_query}, timeout=CHAT_TIMEOUT)
    return reply["answer"], reply["query"]
