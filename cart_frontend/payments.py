"""Razorpay Payment Link helpers (server-side only; the secret never reaches the browser)."""

from __future__ import annotations

import os
import time
from typing import Any

import razorpay
from dotenv import load_dotenv

load_dotenv()

LINK_VALIDITY_SECONDS = 20 * 60  # Razorpay requires expire_by to be >= 15 minutes ahead


def _client() -> razorpay.Client:
    key_id = os.getenv("RAZORPAY_KEY_ID")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET")
    if not key_id or not key_secret:
        raise RuntimeError("Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in your .env file.")
    return razorpay.Client(auth=(key_id, key_secret))


def create_payment_link(amount_rupees: float, customer_name: str, phone_number: str, reference_id: str) -> dict[str, Any]:
    """Create a one-time Razorpay payment link and return its id, URL and amount in paise."""
    amount_paise = int(round(float(amount_rupees) * 100))
    if amount_paise < 100:
        raise ValueError("Razorpay requires a minimum payment of Rs. 1.00.")

    payload: dict[str, Any] = {
        "amount": amount_paise,
        "currency": "INR",
        "accept_partial": False,
        "reference_id": reference_id[:40],
        "description": "Shopping cart purchase",
        "expire_by": int(time.time()) + LINK_VALIDITY_SECONDS,
        "customer": {"name": customer_name},
        "notify": {"sms": False, "email": False},
        "reminder_enable": False,
    }
    if len(phone_number) == 10:  # Razorpay validates the contact format, so only send a clean Indian number
        payload["customer"]["contact"] = f"+91{phone_number}"

    link = _client().payment_link.create(payload)
    return {"id": link["id"], "url": link["short_url"], "amount_paise": amount_paise}


def check_payment_link(link_id: str, expected_amount_paise: int) -> dict[str, Any]:
    """Ask Razorpay for the link's real status. Returns status and, when paid, the payment_id."""
    link = _client().payment_link.fetch(link_id)
    status = link.get("status")  # created | partially_paid | paid | cancelled | expired
    result: dict[str, Any] = {"status": status, "payment_id": None}
    if status == "paid":
        if int(link.get("amount_paid", 0)) != int(expected_amount_paise):
            result["status"] = "amount_mismatch"
            return result
        captured = [p for p in link.get("payments", []) if p.get("status") in ("captured", "authorized")]
        result["payment_id"] = captured[0]["payment_id"] if captured else None
    return result
