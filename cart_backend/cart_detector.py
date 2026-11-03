"""
OpenVINO cart detection core (shared by the Streamlit app and the detect_video.py CLI).

Everything in here is taken verbatim from the validated vspro2 project:
  * letterbox / preprocess, YOLOv8 raw-output decode + per-class NMS
  * cleaning rules: resolution-independent size filter, same-class containment suppression,
    cross-class suppression
  * Tracker with single-instance slots, min-hits and long memory
  * frame_reader thread (prefetches frames while the model runs)
plus `CartAnalyzer`, which chains detector -> cleaning -> tracker -> motion -> add/remove event engine
exactly like detect_video.py does, one frame at a time.
"""
import queue
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import os
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------------------
# CONFIG  (edit here or override on the command line)
# ----------------------------------------------------------------------------------------
FALLBACK_NAMES = ["book", "calculator", "chess_board", "chess_box", "gum",
                  "happydent", "pen", "unomax_pen", "watch"]  # order printed by your validation run

# Classes where the cart can only hold ONE item.
SINGLE_INSTANCE = ["calculator", "chess_board", "chess_box", "book"]

# Minimum box size as a FRACTION of the frame area.
# Converted from your notebook's pixel areas on a 640x640 image (area / 409,600).
MIN_AREA_FRAC = {
    "chess_board": 0.073, "chess_box": 0.061, "book": 0.049, "calculator": 0.029,
    "watch": 0.0073, "happydent": 0.0061, "gum": 0.0044, "unomax_pen": 0.0015, "pen": 0.0015,
}

CLASS_COLORS = {  # BGR
    "chess_board": (180, 105, 255), "chess_box": (255, 20, 147), "book": (0, 165, 255),
    "calculator": (255, 0, 0), "watch": (0, 255, 0), "happydent": (0, 255, 255),
    "gum": (255, 255, 255), "unomax_pen": (0, 140, 255), "pen": (0, 0, 255),
}


# ----------------------------------------------------------------------------------------
# Model loading / preprocessing
# ----------------------------------------------------------------------------------------
def load_names(model_path: Path):
    meta = model_path.parent / "metadata.yaml" if model_path.suffix == ".xml" else model_path / "metadata.yaml"
    if meta.exists():
        import yaml
        names = yaml.safe_load(meta.read_text()).get("names")
        if isinstance(names, dict):
            return [names[k] for k in sorted(names, key=int)]
        if isinstance(names, list):
            return names
    print("[warn] metadata.yaml not found, using FALLBACK_NAMES")
    return FALLBACK_NAMES


def find_xml(model_arg: str) -> Path:
    p = Path(model_arg)
    if p.is_dir():
        xmls = sorted(p.glob("*.xml"))
        if not xmls:
            raise SystemExit(f"No .xml found in {p}")
        return xmls[0]
    return p


def letterbox(img, size, color=114):
    h, w = img.shape[:2]
    r = min(size[0] / h, size[1] / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size[0] - nh) // 2, (size[1] - nw) // 2
    out = np.full((size[0], size[1], 3), color, np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, left, top


def preprocess(frame, size):
    img, r, padx, pady = letterbox(frame, size)
    blob = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(blob), r, padx, pady


# ----------------------------------------------------------------------------------------
# Decode + cleaning
# ----------------------------------------------------------------------------------------
def iou_one_to_many(a, b):
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def inter_and_areas(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy, (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])


def nms(boxes, scores, iou_thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        order = order[1:][iou_one_to_many(boxes[i], boxes[order[1:]]) <= iou_thr]
    return keep


def decode(output, r, padx, pady, orig_w, orig_h, conf_thr, iou_thr):
    """YOLOv8 raw output (1, 4+nc, N) -> list of dets after per-class NMS, in original frame pixels."""
    pred = np.asarray(output).reshape(output.shape[-2:])
    if pred.shape[0] < pred.shape[1]:  # (4+nc, N) -> (N, 4+nc)
        pred = pred.T
    scores_all = pred[:, 4:]
    cls_ids = scores_all.argmax(1)
    scores = scores_all[np.arange(len(pred)), cls_ids]
    m = scores >= conf_thr
    pred, cls_ids, scores = pred[m], cls_ids[m], scores[m]
    if len(pred) == 0:
        return []
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - padx) / r).clip(0, orig_w)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pady) / r).clip(0, orig_h)

    dets = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        for k in nms(boxes[idx], scores[idx], iou_thr):
            dets.append({"cls": int(c), "score": float(scores[idx][k]), "box": boxes[idx][k].astype(np.float64)})
    return dets


def clean_detections(dets, names, frame_area, args):
    """Returns (kept, dropped). dropped items carry a 'reason' for debugging."""
    kept, dropped = [], []

    # 1. size filter relative to frame
    stage = []
    for d in dets:
        b = d["box"]
        frac = (b[2] - b[0]) * (b[3] - b[1]) / frame_area
        if frac < MIN_AREA_FRAC.get(names[d["cls"]], 0.0) * args.min_area_scale:
            dropped.append({**d, "reason": "too_small"})
        else:
            stage.append(d)

    # 2. same-class containment suppression (partial box inside a bigger box)
    stage.sort(key=lambda d: -(d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))  # big first
    alive = []
    for d in stage:
        contained = False
        for big in alive:
            if big["cls"] != d["cls"]:
                continue
            inter, a_big, a_d = inter_and_areas(big["box"], d["box"])
            if inter / (min(a_big, a_d) + 1e-9) > args.contain_thr:
                contained = True
                break
        (dropped.append({**d, "reason": "inside_same_class_box"}) if contained else alive.append(d))

    # 3. cross-class suppression: same object, two labels -> keep higher score
    alive.sort(key=lambda d: -d["score"])
    for d in alive:
        clash = False
        for k in kept:
            if k["cls"] != d["cls"] and iou_one_to_many(k["box"], d["box"][None])[0] > args.cross_iou:
                clash = True
                break
        (dropped.append({**d, "reason": "cross_class_overlap"}) if clash else kept.append(d))
    return kept, dropped


# ----------------------------------------------------------------------------------------
# Tracker: persistent IDs, one slot for single-instance classes
# ----------------------------------------------------------------------------------------
class Track:
    def __init__(self, track_id, cls, box, score, frame_idx):
        self.id = track_id
        self.cls, self.box, self.score = cls, box.copy(), score
        self.hits, self.missed, self.first, self.last = 1, 0, frame_idx, frame_idx
        self.seen_now = True


class Tracker:
    def __init__(self, names, single, iou_thr=0.2, min_hits=3, max_age=45, smooth=0.6):
        self.names, self.single = names, set(single)
        self.iou_thr, self.min_hits, self.max_age, self.smooth = iou_thr, min_hits, max_age, smooth
        self._next_id = 1
        self.tracks = []
        self.ever_confirmed = {}              # track id -> class, for tracks that reached min_hits
        self.peak = defaultdict(int)          # peak simultaneous confirmed tracks per class
        self.first_seen = {}                  # track id -> frame

    def _create_track(self, cls, box, score, frame_idx):
        track = Track(self._next_id, cls, box, score, frame_idx)
        self._next_id += 1
        return track

    def drop(self, track_id):
        """forget a track (used when the item was removed from the cart)"""
        self.tracks = [t for t in self.tracks if t.id != track_id]

    def _update(self, t, d, frame_idx):
        a = self.smooth
        t.box = a * d["box"] + (1 - a) * t.box if t.hits > 1 else d["box"].copy()
        t.score, t.hits, t.missed, t.last, t.seen_now = d["score"], t.hits + 1, 0, frame_idx, True

    def update(self, dets, frame_idx):
        for t in self.tracks:
            t.seen_now = False
        by_cls = defaultdict(list)
        for d in dets:
            by_cls[d["cls"]].append(d)

        for c in set(by_cls) | {t.cls for t in self.tracks}:
            cname, cdets = self.names[c], by_cls.get(c, [])
            ctracks = [t for t in self.tracks if t.cls == c]

            if cname in self.single:
                # ONE slot. Pick the detection that is confident AND consistent with the existing track.
                if cdets:
                    def rank(d):
                        cont = iou_one_to_many(ctracks[0].box, d["box"][None])[0] if ctracks else 0.0
                        return d["score"] + 0.5 * cont
                    best = max(cdets, key=rank)
                    if ctracks:
                        self._update(ctracks[0], best, frame_idx)
                    else:
                        self.tracks.append(self._create_track(c, best["box"], best["score"], frame_idx))
                continue

            # multi-instance: greedy IoU matching, best score first
            free = set(range(len(ctracks)))
            for d in sorted(cdets, key=lambda d: -d["score"]):
                best_i, best_iou = None, self.iou_thr
                for i in free:
                    v = iou_one_to_many(ctracks[i].box, d["box"][None])[0]
                    if v > best_iou:
                        best_i, best_iou = i, v
                if best_i is None:
                    self.tracks.append(self._create_track(c, d["box"], d["score"], frame_idx))
                else:
                    free.discard(best_i)
                    self._update(ctracks[best_i], d, frame_idx)

        for t in self.tracks:
            if not t.seen_now:
                t.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]

        visible = [t for t in self.tracks if t.seen_now and t.hits >= self.min_hits]
        per_cls = Counter(t.cls for t in visible)
        for t in self.tracks:
            if t.hits >= self.min_hits and t.id not in self.ever_confirmed:
                self.ever_confirmed[t.id] = t.cls
                self.first_seen[t.id] = frame_idx
        for c, n in per_cls.items():
            self.peak[c] = max(self.peak[c], n)
        return visible



# ----------------------------------------------------------------------------------------
# Video reader thread (4K decoding is the real bottleneck, so prefetch in the background)
# ----------------------------------------------------------------------------------------
def frame_reader(cap, q, stride, stop):
    i = 0
    while not stop.is_set():
        ok = cap.grab()
        if not ok:
            break
        if i % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            while not stop.is_set():
                try:
                    q.put((i, frame), timeout=0.2)
                    break
                except queue.Full:
                    pass
        i += 1
    while True:  # always leave a sentinel so the consumer never blocks
        try:
            q.put(None, timeout=0.2)
            return
        except queue.Full:
            if stop.is_set():
                return



# ----------------------------------------------------------------------------------------
# Settings (defaults are exactly the detect_video.py command-line defaults)
# ----------------------------------------------------------------------------------------
@dataclass
class Settings:
    device: str = "AUTO"            # AUTO, CPU, GPU (Intel iGPU/Arc) or NPU
    imgsz: int = 640
    conf: float = 0.30
    iou: float = 0.5                # per-class NMS IoU
    contain_thr: float = 0.7        # drop same-class box if this fraction lies inside a bigger one
    cross_iou: float = 0.85         # drop lower-score box when two classes overlap this much
    min_area_scale: float = 1.0     # multiplies every MIN_AREA_FRAC value
    single: list = field(default_factory=lambda: list(SINGLE_INSTANCE))
    min_hits: int = 3
    max_age_s: float = 10.0         # seconds a hidden item stays remembered by the tracker
    settle_s: float = 0.8           # item must stand still this long to count as ADDED
    gone_s: float = 1.2             # item must be undetected this long to count as REMOVED
    quiet_s: float = 0.5            # its spot must be motion-free this long before REMOVED
    motion_thr: float = 0.06        # fraction of a box that must change to count as moving
    initial_s: float = 0.0          # items settled within this many seconds are logged INITIAL
    stride: int = 1                 # process every Nth frame


# ----------------------------------------------------------------------------------------
# OpenVINO model wrapper (compile once, reuse for every video)
# ----------------------------------------------------------------------------------------
class OpenVinoModel:
    def __init__(self, model, device="AUTO", imgsz=640, num_threads=None):
        import openvino as ov

        model = Path(model)
        self.xml = find_xml(str(model))
        if not self.xml.is_file():
            raise FileNotFoundError(f"OpenVINO model not found: {self.xml}")
        if not self.xml.with_suffix(".bin").is_file():
            raise FileNotFoundError(f"OpenVINO weights not found: {self.xml.with_suffix('.bin')}")
        self.names = load_names(model)
        core = ov.Core()

        if num_threads is None:
            model_threads_env = os.getenv("MODEL_THREADS")
            if model_threads_env:
                try:
                    num_threads = int(model_threads_env)
                except ValueError:
                    num_threads = None
            if num_threads is None:
                max_concurrent = int(os.getenv("MAX_CONCURRENT_VIDEOS", "2"))
                cpu_count = os.cpu_count() or 4
                num_threads = max(1, cpu_count // max(1, max_concurrent))

        compile_config = {"PERFORMANCE_HINT": "LATENCY"}
        if num_threads:
            compile_config["INFERENCE_NUM_THREADS"] = int(num_threads)
        self.num_threads = num_threads
        self.compiled = core.compile_model(str(self.xml), device, compile_config)
        shp = self.compiled.input(0).shape
        self.size = (int(shp[2]), int(shp[3])) if len(shp) == 4 and shp[2] > 0 else (imgsz, imgsz)
        self.out_node = self.compiled.output(0)

    def detect(self, frame, conf, iou):
        """Raw decoded detections (before the cleaning rules), in original-frame pixels."""
        H, W = frame.shape[:2]
        blob, r, padx, pady = preprocess(frame, self.size)
        raw = self.compiled(blob)[self.out_node]
        return decode(raw, r, padx, pady, W, H, conf, iou)


# ----------------------------------------------------------------------------------------
# Per-video analyzer: detector -> cleaning -> tracker -> motion -> ADDED / REMOVED events
# (same order of operations as detect_video.py main loop)
# ----------------------------------------------------------------------------------------
class CartAnalyzer:
    def __init__(self, model: OpenVinoModel, settings: Settings, fps: float):
        from cart_events import MotionDetector

        for s in settings.single:
            if s not in model.names:
                raise ValueError(f"single-instance class '{s}' is not in the model classes: {model.names}")
        self.model, self.cfg = model, settings
        self.names = model.names
        self.fps = fps or 30.0
        self.fps_eff = self.fps / settings.stride
        self.tracker = Tracker(self.names, settings.single, min_hits=settings.min_hits,
                               max_age=int(settings.max_age_s * self.fps_eff))
        self.motion = MotionDetector(self.fps_eff)
        self.engine = None
        self.drop_stats = Counter()
        self.conflict_frames = defaultdict(list)

    def process(self, frame, idx):
        """Feed one frame (idx = index in the ORIGINAL video, counting skipped frames)."""
        from cart_events import EventEngine

        H, W = frame.shape[:2]
        c = self.cfg
        if self.engine is None:
            self.engine = EventEngine(self.names, self.fps_eff, W, H, settle_s=c.settle_s, gone_s=c.gone_s,
                                      quiet_s=c.quiet_s, motion_thr=c.motion_thr,
                                      initial_s=c.initial_s, min_hits=c.min_hits)
        dets = self.model.detect(frame, c.conf, c.iou)
        kept, dropped = clean_detections(dets, self.names, W * H, c)
        for d in dropped:
            self.drop_stats[(self.names[d["cls"]], d["reason"])] += 1
        for cls, k in Counter(d["cls"] for d in kept).items():
            if k > 1 and self.names[cls] in c.single:
                self.conflict_frames[self.names[cls]].append(idx)
        self.tracker.update(kept, idx)
        self.motion.update(frame)
        self.engine.update(self.tracker.tracks, self.motion, idx / self.fps, self.tracker)

    def cart(self):
        """Items currently in the cart: {class_name: quantity}."""
        return dict(self.engine.cart()) if self.engine is not None else {}

    def events(self):
        return list(self.engine.events) if self.engine is not None else []

    def finish(self):
        """Returns (events, final_cart). Call once after the last frame."""
        if self.engine is None:
            return [], {}
        return self.engine.finish(), dict(self.engine.cart())

    def unresolved(self):
        """Items that were in the cart but not visible at the end (could not be decided)."""
        if self.engine is None:
            return []
        return [self.names[i.cls] for i in self.engine.items.values() if i.state == "MISSING"]

    def inventory(self):
        inv = {}
        for c in range(len(self.names)):
            peak = self.tracker.peak.get(c, 0)
            if peak:
                uniq = sum(1 for k in self.tracker.ever_confirmed.values() if k == c)
                inv[self.names[c]] = {"count": 1 if self.names[c] in self.cfg.single else peak,
                                      "peak_simultaneous": peak, "unique_tracks": uniq}
        return inv
