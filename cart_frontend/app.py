"""Customer home: chat, video-cart detection, and signed-in shopping features."""

from __future__ import annotations

import hashlib
import io
import os
import time

import streamlit as st
from gtts import gTTS

import api_client as db
import payments
from api_client import get_shopping_assistant_response
from video_client import VideoCartClient


st.set_page_config(page_title="Shopping Cart Assistant", page_icon="🛒", layout="wide")


@st.cache_resource
def get_video_cart_manager() -> VideoCartClient:
    """Return one process-wide client for the backend's video analysis."""
    return VideoCartClient()


def money(amount: float) -> str:
    """Format a numeric value consistently for the small-shop user interface."""
    return f"Rs. {amount:,.2f}"


def clear_customer_session() -> None:
    """Remove only customer-authentication keys from the active Streamlit session."""
    for key in ("customer_id", "customer_name", "customer_phone"):
        st.session_state.pop(key, None)


def render_customer_dashboard() -> None:
    """Render offers, recommendations, and history for a signed-in customer."""
    customer_id = st.session_state.get("customer_id")
    if not customer_id:
        st.info("Log in to see your personal offers, recommendations, and order history.")
        return

    name = st.session_state.get("customer_name", "Customer")
    st.subheader(f"Hello, {name}")
    offers, recommendations, orders = (
        db.get_active_offers(customer_id),
        db.get_recommendations(customer_id),
        db.get_customer_orders(customer_id),
    )

    offers_tab, recommendations_tab, history_tab = st.tabs(
        ["Your offers", "Recommended for you", "Your past orders"]
    )
    with offers_tab:
        if not offers:
            st.caption("No active offers match your purchase history today.")
        for offer in offers:
            value = (
                f"{offer['discount_value']:.0f}% off"
                if offer["discount_type"] == "percentage"
                else f"{money(float(offer['discount_value']))} off"
            )
            target = offer.get("target_category") or offer.get("target_product_name") or "Store-wide"
            st.success(f"{offer['offer_text']} - {value} ({target}, until {offer['valid_until']})")

    with recommendations_tab:
        if not recommendations:
            st.caption("Make your first purchase to receive recommendations.")
        else:
            st.dataframe(
                [
                    {
                        "Product": item["name"],
                        "Category": item["category"],
                        "Price": money(float(item["price"])),
                        "In stock": item["stock_quantity"],
                        "Unit": item["unit"],
                    }
                    for item in recommendations
                ],
                hide_index=True,
                use_container_width=True,
            )

    with history_tab:
        if not orders:
            st.caption("You have not placed an order yet.")
        for order in orders:
            with st.expander(f"Order #{order['id']} - {order['order_date'][:10]} - {money(float(order['total_amount']))}"):
                st.dataframe(
                    [
                        {
                            "Product": item["product_name"],
                            "Qty": item["quantity"],
                            "Price at sale": money(float(item["price_at_sale"])),
                            "Subtotal": money(float(item["subtotal"])),
                        }
                        for item in order["items"]
                    ],
                    hide_index=True,
                    use_container_width=True,
                )


def start_payment(job_id: str, detected_items: list[dict]) -> None:
    """Price the cart from the DB, create a Razorpay payment link and remember it."""
    try:
        items = [{"product_id": i["product_id"], "quantity": i["quantity"]} for i in detected_items]
        total = db.quote_total(items)  # also checks stock before taking money
        link = payments.create_payment_link(
            total,
            st.session_state.get("customer_name", "Customer"),
            st.session_state.get("customer_phone", ""),
            reference_id=f"cart-{st.session_state.customer_id}-{job_id}",
        )
        st.session_state.pending_payments[job_id] = {**link, "items": items}
        st.rerun(scope="fragment")
    except ValueError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Could not start the payment: {error}")


def pay_offline(job_id: str, detected_items: list[dict]) -> None:
    """Place the order directly with offline payment (pay at counter)."""
    try:
        items = [{"product_id": i["product_id"], "quantity": i["quantity"]} for i in detected_items]
        total = db.quote_total(items)  # checks stock before placing
        offline_ref = f"offline-{st.session_state.customer_id}-{job_id}-{int(time.time())}"
        order = db.place_order(
            st.session_state.customer_id,
            items,
            payment_id=offline_ref,
            payment_link_id="offline",
        )
        st.session_state.checked_out_job_ids.append(job_id)
        st.session_state.pending_payments.pop(job_id, None)
        st.session_state[f"order_{job_id}"] = order["id"]
        st.success(f"Offline order #{order['id']} placed for {money(float(order['total_amount']))}. Please pay at the counter.")
        st.rerun(scope="fragment")
    except ValueError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Could not place offline order: {error}")


def handle_pending_payment(job_id: str) -> None:
    """Show the pay link and, once Razorpay reports it paid, save the order."""
    pending = st.session_state.pending_payments[job_id]
    st.info(f"Pay {money(pending['amount_paise'] / 100)} to complete your purchase.")
    st.link_button("Open Razorpay payment page", pending["url"], type="primary")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Cancel payment", key=f"cancel_{job_id}", use_container_width=True):
            st.session_state.pending_payments.pop(job_id, None)
            st.rerun(scope="fragment")
            return
    with btn_col2:
        if st.button("Pay Offline instead", key=f"pay_offline_pending_{job_id}", use_container_width=True):
            st.session_state.pending_payments.pop(job_id, None)
            pay_offline(job_id, pending["items"])
            return
    try:
        result = payments.check_payment_link(pending["id"], pending["amount_paise"])
    except Exception as error:
        st.warning(f"Could not check the payment status yet: {error}")
        return

    if result["status"] == "paid" and result["payment_id"]:
        try:
            order = db.place_order(
                st.session_state.customer_id, pending["items"],
                payment_id=result["payment_id"], payment_link_id=pending["id"],
            )
            st.session_state.checked_out_job_ids.append(job_id)
            st.session_state.pending_payments.pop(job_id, None)
            st.session_state[f"order_{job_id}"] = order["id"]
            st.success(f"Payment received. Order #{order['id']} placed for {money(float(order['total_amount']))}.")
        except ValueError as error:
            st.error(f"Payment {result['payment_id']} was received but the order failed: {error}. Please contact the store.")
    elif result["status"] in {"expired", "cancelled", "amount_mismatch"}:
        st.session_state.pending_payments.pop(job_id, None)
        st.error(f"Payment {result['status'].replace('_', ' ')}. Please try again.")
    else:
        st.caption("Waiting for payment... this page updates automatically.")


for state_key, default_value in {
    "messages": [],
    "video_job_id": None,
    "video_fingerprint": None,
    "checked_out_job_ids": [],
    "pending_payments": {},  # job_id -> {"link_id", "url", "amount_paise", "items"}
}.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value

manager = get_video_cart_manager()

with st.sidebar:
    cart_name = os.getenv("CART_NAME", "Cart 1")
    st.subheader(f"🛒 {cart_name}")
    st.divider()
    st.header("Account")
    if st.session_state.get("customer_id"):
        st.success(f"Signed in as {st.session_state.get('customer_name', 'customer')}")
        if st.button("Customer logout", use_container_width=True):
            clear_customer_session()
            st.rerun()
    else:
        st.info("Shopping chat and video detection work for guests.")
        st.page_link("pages/1_Login_Signup.py", label="Login or create an account", icon="👤")
    st.page_link("pages/2_Admin.py", label="Admin", icon="🔐")

st.title("Shopping Cart Assistant")
st.caption("Ask about the store, upload a cart video, or sign in for a personal shopping view.")

render_customer_dashboard()

st.divider()
video_column, cart_column = st.columns((1, 1), gap="large")
with video_column:
    st.subheader("Cart video")
    uploaded_video = st.file_uploader(
        "Upload a shopping-cart video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Processing begins automatically. Uploaded video files are removed after analysis.",
    )
    if uploaded_video is not None:
        st.video(uploaded_video)
        video_bytes = uploaded_video.getvalue()
        fingerprint = hashlib.sha256(video_bytes).hexdigest()
        if fingerprint != st.session_state.video_fingerprint:
            manager.cancel_job(st.session_state.video_job_id)
            try:
                st.session_state.video_job_id = manager.start_job(video_bytes, uploaded_video.name)
                st.session_state.video_fingerprint = fingerprint
            except Exception as error:
                st.error(f"Could not start video analysis: {error}")
    else:
        st.session_state.video_fingerprint = None


cart_placeholder = cart_column.empty()


@st.fragment(run_every="1s")
def render_live_cart(job_id: str | None) -> None:
    """Refresh live detection/cart state without rerunning the chat interface."""
    with cart_placeholder.container():
        st.subheader("Live cart total")
        state = manager.snapshot(job_id)
        if state is None:
            st.info("Upload a video to start detecting cart items.")
            return

        status = state["status"]
        if status in {"Queued", "Processing", "Cancelling"}:
            st.progress(state["progress"], text=f"{status}: {state['processed_frames']} frames analysed")
        elif status == "Completed":
            st.success(f"Analysis complete: {state['processed_frames']} frames analysed")
        elif status == "Cancelled":
            st.warning("Analysis was cancelled because a newer video was uploaded.")
        else:
            st.error(f"Video analysis failed: {state['error'] or 'Unknown error'}")

        st.metric("Current total", money(float(state["total"])))
        if state["items"]:
            st.dataframe(
                [
                    {
                        "Product": item["product"],
                        "Detected as": item["label"],
                        "Qty": item["quantity"],
                        "Unit price": money(float(item["unit_price"])),
                        "Subtotal": money(float(item["subtotal"])),
                    }
                    for item in state["items"]
                ],
                hide_index=True,
                use_container_width=True,
            )
        elif status in {"Queued", "Processing", "Cancelling"}:
            st.caption("Waiting for the first item to be placed in the cart.")

        if state["events"]:
            with st.expander(f"Add / remove log ({len(state['events'])} events)"):
                st.dataframe(
                    [
                        {
                            "Time (s)": event["time_s"],
                            "Event": event["event"],
                            "Item": event["item"],
                            "Detail": (
                                f"entered from {event['entered_from']}"
                                if event.get("entered_from")
                                else f"left toward {event.get('left_toward', '?')} "
                                f"({event.get('confidence', '')})"
                            ),
                        }
                        for event in state["events"]
                    ],
                    hide_index=True,
                    use_container_width=True,
                )

        if state["unresolved"]:
            st.info(
                "Not visible at the end of the video (still counted in the cart): "
                + ", ".join(sorted(state["unresolved"]))
            )

        if state["unmatched_labels"]:
            st.warning(
                "No database price was found for: "
                + ", ".join(sorted(state["unmatched_labels"]))
                + ". These labels are excluded from the total."
            )

        if status == "Completed" and state["items"]:
            if not st.session_state.get("customer_id"):
                st.info("Log in to finalize this detected cart as an order.")
            elif job_id in st.session_state.checked_out_job_ids:
                order_id = st.session_state.get(f"order_{job_id}")
                if order_id:
                    st.success(f"Order #{order_id} has been saved. Please pay at the counter if paying offline.")
                else:
                    st.success("This detected cart has already been saved as an order.")
            elif job_id in st.session_state.pending_payments:
                handle_pending_payment(job_id)
            else:
                pay_col1, pay_col2 = st.columns(2)
                with pay_col1:
                    if st.button("Pay with Razorpay", key=f"checkout_{job_id}", type="primary", use_container_width=True):
                        start_payment(job_id, state["items"])
                with pay_col2:
                    if st.button("Pay Offline", key=f"offline_{job_id}", use_container_width=True):
                        pay_offline(job_id, state["items"])


render_live_cart(st.session_state.video_job_id)

st.divider()
st.subheader("Store assistant")
st.caption("You can keep chatting while a cart video is processing.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("audio") is not None:
            st.audio(message["audio"], format="audio/mp3")

if prompt := st.chat_input("Example: Is pen available?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Checking the aisles..."):
            try:
                malayalam_text, _sql_query = get_shopping_assistant_response(prompt)
                st.markdown(malayalam_text)
                audio_bytes = None
                if not malayalam_text.startswith("**") and not malayalam_text.startswith("I'm sorry"):
                    tts = gTTS(text=malayalam_text, lang="ml")
                    audio_buffer = io.BytesIO()
                    tts.write_to_fp(audio_buffer)
                    audio_bytes = audio_buffer.getvalue()
                    st.audio(audio_bytes, format="audio/mp3")
                st.session_state.messages.append(
                    {"role": "assistant", "content": malayalam_text, "audio": audio_bytes}
                )
            except Exception as error:
                error_message = f"Sorry, I encountered an error: {error}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
