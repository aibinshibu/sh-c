"""Background OpenVINO video processing for the Streamlit shopping-cart app.

The detection pipeline is the validated one from the vspro2 project (see cart_detector.py and
cart_events.py): OpenVINO inference -> cleaning rules -> tracker -> motion detection ->
ADDED / REMOVED events.  The cart that gets billed is the cart the event engine ends up with,
so an item that was picked up and put back, or taken out again, is not charged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import queue
import shutil
import tempfile
import threading
import uuid

from assistant import get_product_price_for_detector_label
from cart_detector import CartAnalyzer, OpenVinoModel, Settings, frame_reader


@dataclass
class VideoCartJob:
    job_id: str
    filename: str
    status: str = "Queued"
    progress: float = 0.0
    processed_frames: int = 0
    total_frames: int = 0
    items: list[dict] = field(default_factory=list)
    total: float = 0.0
    unmatched_labels: list[str] = field(default_factory=list)
    price_queries: dict[str, str] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)  # ADDED / REMOVED / INITIAL log
    unresolved: list[str] = field(default_factory=list)  # in cart earlier, not visible at the end
    error: str | None = None
    cancel_requested: bool = False


class VideoCartManager:
    """Keeps long-running inference outside Streamlit's request thread."""

    def __init__(self, model_path: str | Path, settings: Settings | None = None):
        # model_path is the best.xml file; best.bin and metadata.yaml sit next to it.
        self.model_path = Path(model_path)
        self.settings = settings or Settings()
        self.max_concurrent_videos = max(1, int(os.getenv("MAX_CONCURRENT_VIDEOS", "2")))
        self._jobs: dict[str, VideoCartJob] = {}
        self._lock = threading.RLock()
        self._model_pool: queue.Queue[OpenVinoModel] = queue.Queue()
        self._model_lock = threading.Lock()
        self._total_models_loaded = 0
        # label -> (product record or None, sql query); products do not change between videos
        self._price_cache: dict[str, tuple[dict | None, str]] = {}

    @property
    def models_loaded(self) -> int:
        return self._total_models_loaded

    @property
    def videos_running(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == "Processing")

    @property
    def videos_queued(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == "Queued")

    def load_models(self) -> int:
        """Create MAX_CONCURRENT_VIDEOS OpenVinoModel instances in the pool."""
        with self._model_lock:
            while not self._model_pool.empty():
                try:
                    self._model_pool.get_nowait()
                except queue.Empty:
                    break
            loaded = 0
            for _ in range(self.max_concurrent_videos):
                model = OpenVinoModel(self.model_path, self.settings.device, self.settings.imgsz)
                self._model_pool.put(model)
                loaded += 1
            self._total_models_loaded = loaded
            return loaded

    def _get_model(self) -> OpenVinoModel:
        """Backward-compatible model getter: ensures models are loaded and returns one."""
        with self._model_lock:
            if self._total_models_loaded == 0:
                self.load_models()
            model = self._model_pool.get()
            self._model_pool.put(model)
            return model

    def _borrow_model(self, job_id: str) -> OpenVinoModel | None:
        """Acquire one model from the pool, waiting until a slot opens or the job is cancelled."""
        with self._model_lock:
            if self._total_models_loaded == 0:
                self.load_models()
        while not self._was_cancelled(job_id):
            try:
                return self._model_pool.get(timeout=0.2)
            except queue.Empty:
                continue
        return None

    def start_job(self, video_bytes: bytes, filename: str) -> str:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"OpenVINO model was not found: {self.model_path.name}")
        if not self.model_path.with_suffix(".bin").is_file():
            raise FileNotFoundError(f"OpenVINO weights were not found: {self.model_path.with_suffix('.bin').name}")
        if not video_bytes:
            raise ValueError("The uploaded video is empty.")

        suffix = Path(filename).suffix.lower()
        if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            suffix = ".mp4"
        job_id = uuid.uuid4().hex
        job = VideoCartJob(job_id=job_id, filename=Path(filename).name)
        working_dir = Path(tempfile.mkdtemp(prefix="shopping_cart_video_"))
        video_path = working_dir / f"upload{suffix}"
        video_path.write_bytes(video_bytes)

        with self._lock:
            self._jobs[job_id] = job

        print(f"[video] Job {job_id[:8]} queued ({Path(filename).name})")

        thread = threading.Thread(
            target=self._process,
            args=(job_id, video_path, working_dir),
            name=f"video-cart-{job_id[:8]}",
            daemon=True,
        )
        thread.start()
        return job_id

    def cancel_job(self, job_id: str | None) -> None:
        if not job_id:
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status in {"Queued", "Processing"}:
                job.cancel_requested = True
                job.status = "Cancelling"

    def snapshot(self, job_id: str | None) -> dict | None:
        if not job_id:
            return None
        with self._lock:
            job = self._jobs.get(job_id)
            return asdict(job) if job else None

    def _was_cancelled(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.cancel_requested if job else False

    def _set_progress(self, job_id: str, frame_number: int, total_frames: int) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.processed_frames = frame_number
            job.total_frames = total_frames
            if total_frames > 0:
                # Reserve the final fraction for price lookup and completion.
                job.progress = min(0.99, frame_number / total_frames)

    def _price_for(self, label: str) -> tuple[dict | None, str]:
        cached = self._price_cache.get(label)
        if cached is not None:
            return cached
        try:
            result = get_product_price_for_detector_label(label)
        except Exception:
            return None, "N/A"  # not cached, so a later video retries the lookup
        self._price_cache[label] = result
        return result

    def _refresh_cart(self, job_id: str, counts: dict[str, int]) -> None:
        """Rebuild the billed items from the cart counts. Labels no longer in the cart drop out."""
        resolved: dict[str, tuple[dict | None, str]] = {}
        for label in counts:
            if self._was_cancelled(job_id):
                return
            resolved[label] = self._price_for(label)

        with self._lock:
            job = self._jobs[job_id]
            items: list[dict] = []
            for label, quantity in counts.items():
                product, sql_query = resolved[label]
                job.price_queries[label] = sql_query
                if product is None:
                    if label not in job.unmatched_labels:
                        job.unmatched_labels.append(label)
                    continue
                items.append({
                    "product_id": product["id"],
                    "label": label,
                    "product": product["name"],
                    "quantity": quantity,
                    "unit_price": product["price"],
                    "subtotal": round(product["price"] * quantity, 2),
                })
            # a label that was unmatched earlier but is no longer in the cart is not a warning any more
            job.unmatched_labels = [label for label in job.unmatched_labels if label in counts]
            job.items = sorted(items, key=lambda item: item["product"].lower())
            job.total = round(sum(item["subtotal"] for item in job.items), 2)

    def _process(self, job_id: str, video_path: Path, working_dir: Path) -> None:
        stop = threading.Event()
        capture = None
        reader: threading.Thread | None = None
        model: OpenVinoModel | None = None
        try:
            model = self._borrow_model(job_id)
            if self._was_cancelled(job_id) or model is None:
                if model is not None:
                    self._model_pool.put(model)
                    model = None
                with self._lock:
                    self._jobs[job_id].status = "Cancelled"
                print(f"[video] Job {job_id[:8]} cancelled while queued")
                return

            with self._lock:
                self._jobs[job_id].status = "Processing"
            print(f"[video] Job {job_id[:8]} started processing")

            try:
                import cv2
            except ImportError as error:
                raise RuntimeError(
                    "OpenCV is unavailable. Reinstall the updated requirements."
                ) from error

            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError("The uploaded video could not be opened.")
            fps = capture.get(cv2.CAP_PROP_FPS) or 30
            total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            stride = max(1, self.settings.stride)

            analyzer = CartAnalyzer(model, self.settings, fps)
            current_counts: dict[str, int] | None = None

            # Decoding (esp. 4K) is the bottleneck, so frames are prefetched in a thread.
            frames: queue.Queue = queue.Queue(maxsize=8)
            reader = threading.Thread(
                target=frame_reader, args=(capture, frames, stride, stop), daemon=True
            )
            reader.start()

            processed = 0
            while True:
                item = frames.get()
                if item is None:
                    break
                frame_index, frame = item
                if self._was_cancelled(job_id):
                    with self._lock:
                        self._jobs[job_id].status = "Cancelled"
                    print(f"[video] Job {job_id[:8]} cancelled while processing")
                    return

                analyzer.process(frame, frame_index)
                processed += 1

                cart_counts = analyzer.cart()
                counts_changed = cart_counts != current_counts
                if counts_changed:
                    current_counts = cart_counts
                    self._refresh_cart(job_id, cart_counts)
                    with self._lock:
                        self._jobs[job_id].events = analyzer.events()

                # A few UI updates per second are enough for a live display
                # and avoid needless lock contention during inference.
                if processed == 1 or processed % 10 == 0 or counts_changed:
                    self._set_progress(job_id, frame_index + 1, total_frames)

            events, final_cart = analyzer.finish()
            self._refresh_cart(job_id, final_cart)

            with self._lock:
                job = self._jobs[job_id]
                job.events = events
                job.unresolved = analyzer.unresolved()
                job.processed_frames = max(job.processed_frames, total_frames)
                job.progress = 1.0
                job.status = "Completed"
            print(f"[video] Job {job_id[:8]} finished (status=Completed)")
        except Exception as error:
            with self._lock:
                job = self._jobs[job_id]
                job.status = "Failed"
                job.error = str(error)
            print(f"[video] Job {job_id[:8]} finished (status=Failed, error={error})")
        finally:
            if model is not None:
                self._model_pool.put(model)
                model = None
            stop.set()
            if reader is not None:
                reader.join(timeout=5)  # never release the capture while the reader may still use it
            if capture is not None:
                capture.release()
            # The uploaded video is no longer needed after inference completes.
            shutil.rmtree(working_dir, ignore_errors=True)
