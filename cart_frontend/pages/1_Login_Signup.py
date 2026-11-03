"""Customer account login and signup page."""

import streamlit as st

import api_client as db


st.set_page_config(page_title="Customer account", page_icon="👤", layout="wide")


def logout() -> None:
    """Clear customer authentication state from the current Streamlit session."""
    st.session_state.pop("customer_id", None)
    st.session_state.pop("customer_name", None)
    st.session_state.pop("customer_phone", None)


st.title("Customer account")
st.caption("Create an account to view personal offers, recommendations, and order history.")

if st.session_state.get("customer_id"):
    st.success(f"You are signed in as {st.session_state.get('customer_name', 'customer')}.")
    left, right = st.columns(2)
    with left:
        if st.button("Go to shopping", type="primary", use_container_width=True):
            st.switch_page("app.py")
    with right:
        if st.button("Log out", use_container_width=True):
            logout()
            st.rerun()
    st.stop()

login_tab, signup_tab = st.tabs(["Log in", "Create account"])

with login_tab:
    with st.form("customer_login_form"):
        phone = st.text_input("Phone number", max_chars=15)
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", type="primary")
    if submitted:
        try:
            customer = db.verify_login(phone, password)
            if customer is None:
                st.error("Incorrect phone number or password.")
            else:
                st.session_state.customer_id = customer["id"]
                st.session_state.customer_name = customer["name"]
                st.session_state.customer_phone = customer["phone_number"]
                st.success("Logged in successfully. You can return to shopping now.")
        except ValueError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Could not log in: {error}")

with signup_tab:
    with st.form("customer_signup_form", clear_on_submit=True):
        name = st.text_input("Full name")
        phone = st.text_input("Phone number", max_chars=15, help="Digits only, 8 to 15 digits.")
        password = st.text_input("Password", type="password", help="At least 6 characters.")
        confirm_password = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create account", type="primary")
    if submitted:
        if password != confirm_password:
            st.error("The passwords do not match.")
        else:
            try:
                customer_id = db.create_customer(name, phone, password)
                st.session_state.customer_id = customer_id
                st.session_state.customer_name = name.strip()
                st.session_state.customer_phone = phone.strip()
                st.success("Account created successfully. You are now logged in.")
            except ValueError as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"Could not create the account: {error}")

if st.button("Back to shopping", use_container_width=True):
    st.switch_page("app.py")
