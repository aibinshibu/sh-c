# Shopping Cart: frontend and backend

Two programs on one computer. Start the backend first.

| Program | Folder | Port | What it holds |
|---|---|---|---|
| Backend (FastAPI) | `cart_backend` | 8000 | OpenVINO model pool, database, Gemini assistant, video jobs |
| Frontend (Streamlit) | `cart_frontend` | 8501 / 8502 | Pages, payment screen, backend client |

## One-time setup (do it in each folder, in its own terminal)

1. Create and activate a virtual environment.
2. `pip install -r requirements.txt`
3. In `cart_backend`, run `python setupdb.py` to create the initial sample database.
   This recreates `supermarket1.db`; back up any data you need before running it.
4. Copy `.env.example` to `.env` and fill it in.
   Both folders need the **same** `API_KEY` value.
   Only the backend needs `GOOGLE_API_KEY`. Only the frontend needs `ADMIN_PASSWORD` and the Razorpay keys.

## Environment Variables

### Backend (`cart_backend/.env`)
- `API_KEY`: Shared secret key required for API access.
- `GOOGLE_API_KEY`: API key for Gemini assistant.
- `MODEL_PATH`: Path to OpenVINO XML model (default: `models/best.xml`).
- `MAX_CONCURRENT_VIDEOS`: Max videos analysed concurrently (default: `2`). Extra jobs wait in `Queued` status.
- `MODEL_THREADS`: Optional CPU thread limit per model instance (default: `max(1, cpu_count // MAX_CONCURRENT_VIDEOS)`).

### Frontend (`cart_frontend/.env`)
- `BACKEND_URL`: URL of the FastAPI backend (e.g. `http://localhost:8000`).
- `API_KEY`: Shared secret matching backend `API_KEY`.
- `ADMIN_PASSWORD`: Password for admin portal.
- `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET`: Razorpay credentials.cd c:\cart\cart_frontend

- `CART_NAME`: Name displayed at top of the sidebar (default: `Cart 1`).

---

## How to Run Both Carts at the Same Time

You have two ways to do this depending on whether you want to use 2 terminals or 3 terminals.

### Method 1: The Quick Way (2 Terminals Total)
You run **one** backend and **one** frontend server, then open **two browser windows**.

1. **Terminal 1 (Backend):**
   ```powershell
   cd c:\cart\cart_backend
   .\.venv\Scripts\Activate.ps1
   python -m uvicorn main:app --port 8000
   ```
2. **Terminal 2 (Frontend):**
   ```powershell
   cd c:\cart\cart_frontend
   .\.venv\Scripts\Activate.ps1
   streamlit run app.py
   ```
3. **Open Two Browser Windows:**
   - Window 1: Go to `http://localhost:8501` (Cart A)
   - Window 2: Open an **Incognito / Private Window** (or a 2nd browser like Edge/Chrome) and go to `http://localhost:8501` (Cart B)
   *(Tip: Using an Incognito window lets each cart log in as a different customer).*
   - Now upload a video on Window 1, and simultaneously upload a video on Window 2. Both process at the same time!

---

### Method 2: Dedicated Named Carts (3 Terminals Total)
If you want two clearly separated carts with their own labels ("Cart 1" and "Cart 2") and ports:

1. **Terminal 1 (Backend):**
   ```powershell
   cd c:\cart\cart_backend
   .\.venv\Scripts\Activate.ps1
   python -m uvicorn main:app --port 8000
   ```
   *(Keep Uvicorn at ONE worker. Video jobs and models are kept in memory).*

2. **Terminal 2 (Cart 1 on port 8501):**
   ```powershell
   cd c:\cart\cart_frontend
   .\.venv\Scripts\Activate.ps1
   $env:CART_NAME="Cart 1"
   streamlit run app.py --server.port 8501
   ```
   Open in browser: **http://localhost:8501**

3. **Terminal 3 (Cart 2 on port 8502):**
   ```powershell
   cd c:\cart\cart_frontend
   .\.venv\Scripts\Activate.ps1
   $env:CART_NAME="Cart 2"
   streamlit run app.py --server.port 8502
   ```
   Open in browser: **http://localhost:8502**

Both carts run side-by-side, talk to the same backend at `http://localhost:8000`, process videos concurrently, and show their cart names in the sidebar.

---

## Health & Status Check

To verify concurrency and see how many models are loaded or jobs are currently running:
* Open **http://localhost:8000/health** in your browser.
* You will see:
  ```json
  {
    "status": "ok",
    "model_found": true,
    "max_concurrent_videos": 2,
    "models_loaded": 2,
    "videos_running": 0,
    "videos_queued": 0
  }
  ```

---

## Good to know

- The database is `cart_backend/supermarket1.db`. It is created locally by `setupdb.py` and is not included in the repository.
- The model files are `cart_backend/models/best.xml`, `best.bin` and `metadata.yaml`. Keep all three together.
- The pages use `api_client.py` in place of the old `db.py`. It has the same function names, so the pages barely changed.
- Chat and price matching call Gemini, and payments call Razorpay, so the computer needs internet.

---

## Adding products that the video detector can recognise

The video detector identifies objects using the class names in `models/metadata.yaml`:

```
book, calculator, chess_board, chess_box, gum, happydent, pen, unomax_pen, watch
```

When a label is detected the backend looks it up **without calling Gemini** by normalising
both the detector label and every product name/alternate-name (lowercase, remove all
non-alphanumeric characters) and finding an exact match.

**Naming rule:** to make detector label `xyz_label` resolve to a product, either:

1. **Name the product** so that its name normalises to the same string.
   - e.g. label `chess_board` → `chessboard` or `chess board` both normalise to `chessboard`. ✓
2. **Add an alternate name** (comma-separated in the `alternate_names` column) that
   normalises to the label.
   - e.g. label `unomax_pen` → product "unomax pen" with `alternate_names = "unomaxpen"` ✓
   - e.g. label `chess_board` → product "chessboard" with `alternate_names = "chess board,chess_board"` ✓

If two products normalise to the same label the detector skips that detection and
prints a warning rather than billing the wrong product.
