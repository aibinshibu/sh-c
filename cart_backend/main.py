"""FastAPI backend for the shopping-cart project.

Holds everything that is not a screen: the OpenVINO model and video analysis, the SQLite
database (db.py), and the Gemini assistant (assistant.py). The Streamlit frontend talks to
it through the routes below, sending the shared secret in the X-API-Key header.

Start it from inside this folder:   uvicorn main:app --port 8000
Use ONE worker (the default): video jobs are kept in memory.
Try every route by hand at:         http://localhost:8000/docs
"""

from __future__ import annotations

import hmac
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional, Union

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from fastapi import APIRouter, Depends, FastAPI, File, Header, HTTPException, UploadFile  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import db  # noqa: E402
from assistant import get_shopping_assistant_response  # noqa: E402
from video_service import VideoCartManager  # noqa: E402

API_KEY = os.getenv("API_KEY", "")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best.xml"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = BASE_DIR / MODEL_PATH

# One manager for the whole backend (video jobs live in memory, so run a single worker).
manager = VideoCartManager(MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEY:
        raise RuntimeError("API_KEY is not set. Add it to cart_backend/.env before starting.")
    # Load the models now so customers do not wait for compilation.
    try:
        loaded = manager.load_models()
        print(f"[backend] OpenVINO models loaded: {loaded}/{manager.max_concurrent_videos} ({MODEL_PATH.name})")
    except Exception as error:  # the backend still starts; the first video job reports the problem
        print(f"[backend] WARNING: models could not be loaded: {error}")
    yield


app = FastAPI(title="Shopping Cart Backend", lifespan=lifespan)


# ----------------------------------------------------------------------------------------
# Security and errors
# ----------------------------------------------------------------------------------------
def require_key(x_api_key: str = Header(default="")) -> None:
    if not API_KEY or not hmac.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Missing or wrong API key.")


@app.exception_handler(ValueError)
async def value_error_handler(request, error: ValueError):
    # db.py raises ValueError for bad input; the pages show this text to the user.
    return JSONResponse(status_code=400, content={"detail": str(error)})


@app.exception_handler(RequestValidationError)
async def request_validation_handler(request, error: RequestValidationError):
    first = error.errors()[0] if error.errors() else {}
    where = ".".join(str(part) for part in first.get("loc", []) if part != "body")
    return JSONResponse(
        status_code=400,
        content={"detail": f"Invalid input{' for ' + where if where else ''}: {first.get('msg', 'bad value')}"},
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request, error: Exception):
    return JSONResponse(status_code=500, content={"detail": str(error) or error.__class__.__name__})


def _json_default(value: Any) -> Any:
    """Turn numpy numbers (from the detector) into plain Python numbers."""
    if hasattr(value, "item"):
        return value.item()
    return str(value)


# ----------------------------------------------------------------------------------------
# Request bodies
# ----------------------------------------------------------------------------------------
class ProductIn(BaseModel):
    name: str
    category: str
    price: float
    shelf_no: int
    row_no: int
    stock_quantity: int
    unit: str = "piece"
    alternate_names: Optional[str] = None


class CustomerIn(BaseModel):
    name: str
    phone_number: str
    password: str


class LoginIn(BaseModel):
    phone_number: str
    password: str


class OfferIn(BaseModel):
    offer_text: str
    target_product_id: Optional[int] = None
    target_category: Optional[str] = None
    discount_value: float
    discount_type: str
    valid_from: str
    valid_until: str
    is_active: Union[bool, int] = True


class ItemsIn(BaseModel):
    items: list[dict[str, Any]]


class OrderIn(BaseModel):
    customer_id: int
    items: list[dict[str, Any]]
    payment_id: Optional[str] = None
    payment_link_id: Optional[str] = None


class ChatIn(BaseModel):
    query: str


# ----------------------------------------------------------------------------------------
# Routes (every route here needs the API key; /health below does not)
# ----------------------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_found": MODEL_PATH.is_file(),
        "max_concurrent_videos": manager.max_concurrent_videos,
        "models_loaded": manager.models_loaded,
        "videos_running": manager.videos_running,
        "videos_queued": manager.videos_queued,
    }


router = APIRouter(dependencies=[Depends(require_key)])
# Plain "def" routes (not async): the database calls block, so FastAPI runs each in a thread.


# --- Products
@router.get("/products")
def list_products(search: str = "", low_stock_threshold: Optional[int] = None):
    return db.get_all_products(search, low_stock_threshold)


@router.get("/products/{product_id}")
def get_product(product_id: int):
    return db.get_product_by_id(product_id)


@router.post("/products")
def create_product(body: ProductIn):
    return db.add_product(
        body.name, body.category, body.price, body.shelf_no, body.row_no,
        body.stock_quantity, body.unit, body.alternate_names,
    )


@router.put("/products/{product_id}")
def change_product(product_id: int, body: ProductIn):
    return db.update_product(
        product_id, body.name, body.category, body.price, body.shelf_no, body.row_no,
        body.stock_quantity, body.unit, body.alternate_names,
    )


@router.delete("/products/{product_id}")
def remove_product(product_id: int):
    return db.delete_product(product_id)


# --- Customers
@router.post("/customers")
def sign_up(body: CustomerIn):
    return db.create_customer(body.name, body.phone_number, body.password)


@router.post("/login")
def log_in(body: LoginIn):
    return db.verify_login(body.phone_number, body.password)


@router.get("/customers/{customer_id}/offers")
def customer_offers(customer_id: int):
    return db.get_active_offers(customer_id)


@router.get("/customers/{customer_id}/recommendations")
def customer_recommendations(customer_id: int, limit: int = 8):
    return db.get_recommendations(customer_id, limit)


@router.get("/customers/{customer_id}/orders")
def customer_orders(customer_id: int):
    return db.get_customer_orders(customer_id)


# --- Offers
@router.get("/offers")
def list_offers():
    return db.get_all_offers()


@router.get("/offers/{offer_id}")
def get_offer(offer_id: int):
    return db.get_offer_by_id(offer_id)


@router.post("/offers")
def create_offer(body: OfferIn):
    return db.add_offer(
        body.offer_text, body.target_product_id, body.target_category, body.discount_value,
        body.discount_type, body.valid_from, body.valid_until, body.is_active,
    )


@router.put("/offers/{offer_id}")
def change_offer(offer_id: int, body: OfferIn):
    return db.update_offer(
        offer_id, body.offer_text, body.target_product_id, body.target_category, body.discount_value,
        body.discount_type, body.valid_from, body.valid_until, body.is_active,
    )


@router.delete("/offers/{offer_id}")
def remove_offer(offer_id: int):
    return db.delete_offer(offer_id)


# --- Orders
@router.post("/orders/quote")
def quote_order(body: ItemsIn):
    return db.quote_total(body.items)  # also checks stock


@router.post("/orders")
def create_order(body: OrderIn):
    return db.place_order(
        body.customer_id, body.items, payment_id=body.payment_id, payment_link_id=body.payment_link_id,
    )


# --- Admin reports
@router.get("/admin/customers")
def admin_customers():
    return db.get_customer_summaries()


@router.get("/admin/sales-summary")
def admin_sales_summary():
    return db.get_sales_summary()


@router.get("/admin/orders")
def admin_orders():
    return db.get_sales_orders()


# --- Chat (Gemini assistant)
@router.post("/chat")
def chat(body: ChatIn):
    answer, query = get_shopping_assistant_response(body.query)
    return {"answer": answer, "query": query}


# --- Video analysis jobs
@router.post("/video-jobs")
def start_video_job(file: UploadFile = File(...)):
    video_bytes = file.file.read()
    return {"job_id": manager.start_job(video_bytes, file.filename or "upload.mp4")}


@router.get("/video-jobs/{job_id}")
def get_video_job(job_id: str):
    snapshot = manager.snapshot(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Unknown video job.")
    return JSONResponse(content=json.loads(json.dumps(snapshot, default=_json_default)))


@router.delete("/video-jobs/{job_id}")
def cancel_video_job(job_id: str):
    manager.cancel_job(job_id)
    return {"ok": True}


app.include_router(router)
