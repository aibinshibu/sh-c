"""
Run your shopping-cart YOLOv8 model locally with OpenVINO and count each item once.

    python detect_video.py --model best.xml --video my_video.mp4

Why this fixes "2 calculators instead of 1"
-------------------------------------------
1. Containment suppression: a partial box sitting inside a bigger box of the same class
   has a LOW IoU with it, so normal NMS keeps both. We drop it using
   intersection / area-of-the-smaller-box instead.
2. Resolution-independent size filter: your notebook used absolute pixel areas
   (calculator >= 12000 px). That was tuned on 640px images; on a 4K video it filters
   almost nothing. Here the limits are a fraction of the frame area.
3. Cross-class suppression: one object labelled as two different classes.
4. Single-instance classes: only one calculator can exist. The tracker keeps ONE
   persistent track for it, so it is also not counted twice when it is hidden by a hand
   and reappears.
5. Tracking + min-hits: a 1-2 frame flicker never becomes an item.

Add / remove events (cart_events.py)
------------------------------------
Motion detection (with camera-shake cancelling) plus the tracker decide whether an item was
ADDED to or REMOVED from the cart. Events are printed, shown on the video, and written to
<output>.events.csv / .events.json.  Disable with --no-events, see the red motion tint with --show-motion.

Use --debug-dir to save frames where a single-instance class had competing boxes, so you
can see exactly which object is being mislabelled (and use them as retraining data).
"""
import argparse
import csv
import json
import queue
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# Shared detection core (same code the Streamlit app uses)
from cart_detector import (  # noqa: E402,F401
    FALLBACK_NAMES, SINGLE_INSTANCE, MIN_AREA_FRAC, CLASS_COLORS,
    load_names, find_xml, letterbox, preprocess, iou_one_to_many, inter_and_areas, nms,
    decode, clean_detections, Track, Tracker, frame_reader,
)

# ----------------------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------------------
def draw_label(img, text, x1, y1, color, scale):
    th = max(1, int(scale * 2))
    (tw, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    y0 = max(y1, h + 8)
    cv2.rectangle(img, (x1, y0 - h - 8), (x1 + tw + 6, y0), color, -1)
    cv2.putText(img, text, (x1 + 3, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th, cv2.LINE_AA)


def draw_tracks(img, tracks, names, s, engine=None):
    scale = max(0.5, img.shape[1] / 1400)
    for t in tracks:
        x1, y1, x2, y2 = (t.box * s).astype(int)
        color = CLASS_COLORS.get(names[t.cls], (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, int(scale * 3)))
        tag = " (placing)" if engine is not None and engine.status(t.id) == "PENDING" else ""
        draw_label(img, f"{names[t.cls]} #{t.id} {t.score:.2f}{tag}", x1, y1, color, scale)
    return img


def draw_events_hud(img, engine, motion, t):
    """cart contents (top-left), motion indicator, and the ADDED / REMOVED banner"""
    scale = max(0.5, img.shape[1] / 1600)
    lines = ["IN CART:"] + ([f"  {n} x{c}" for n, c in sorted(engine.cart().items())] or ["  (empty)"])
    y = int(40 * scale)
    for ln in lines:
        cv2.putText(img, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), int(4 * scale) + 1, cv2.LINE_AA)
        cv2.putText(img, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), int(2 * scale) + 1, cv2.LINE_AA)
        y += int(38 * scale)
    if motion.camera_moving:
        label, col = "CAMERA MOVING", (0, 165, 255)
    elif motion.global_ratio > 0.02:
        label, col = "MOTION", (0, 0, 255)
    else:
        label, col = "still", (0, 160, 0)
    cv2.circle(img, (img.shape[1] - int(150 * scale), int(30 * scale)), int(10 * scale), col, -1)
    cv2.putText(img, label, (img.shape[1] - int(130 * scale), int(38 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                scale * 0.8, col, int(2 * scale) + 1, cv2.LINE_AA)
    by = img.shape[0] - int(60 * scale)
    for _, text, color in engine.popup_lines(t):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale * 1.4, 3)
        x = (img.shape[1] - tw) // 2
        cv2.rectangle(img, (x - 12, by - th - 14), (x + tw + 12, by + 12), color, -1)
        cv2.putText(img, text, (x, by), cv2.FONT_HERSHEY_SIMPLEX, scale * 1.4, (255, 255, 255), 3, cv2.LINE_AA)
        by -= th + 40
    return img


def draw_hud(img, tracker, visible, names):
    scale = max(0.5, img.shape[1] / 1600)
    cnt = Counter(names[t.cls] for t in visible)
    lines = [f"{n}: {c}" for n, c in sorted(cnt.items())] or ["(nothing confirmed)"]
    y = int(40 * scale)
    for ln in lines:
        cv2.putText(img, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), int(4 * scale) + 1, cv2.LINE_AA)
        cv2.putText(img, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), int(2 * scale) + 1, cv2.LINE_AA)
        y += int(38 * scale)
    return img


def draw_debug(frame, kept, dropped, names, s):
    img = frame.copy()
    scale = max(0.5, img.shape[1] / 1400)
    for d in dropped:
        x1, y1, x2, y2 = (d["box"] * s).astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (150, 150, 150), 2)
        draw_label(img, f"X {names[d['cls']]} {d['score']:.2f} {d['reason']}", x1, y2 - 4, (150, 150, 150), scale * 0.7)
    for d in kept:
        x1, y1, x2, y2 = (d["box"] * s).astype(int)
        col = CLASS_COLORS.get(names[d["cls"]], (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
        draw_label(img, f"{names[d['cls']]} {d['score']:.2f}", x1, y1, col, scale)
    return img


# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="folder containing best.xml, or the best.xml file itself")
    ap.add_argument("--video", required=True)
    ap.add_argument("--output", default="cleaned_cart_video.mp4")
    ap.add_argument("--device", default="AUTO", help="AUTO, CPU, GPU (Intel iGPU/Arc) or NPU")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--iou", type=float, default=0.5, help="per-class NMS IoU (ultralytics default 0.7 is looser)")
    ap.add_argument("--contain-thr", type=float, default=0.7, help="drop same-class box if this fraction of it lies inside a bigger one")
    ap.add_argument("--cross-iou", type=float, default=0.85, help="drop lower-score box when two classes overlap this much")
    ap.add_argument("--min-area-scale", type=float, default=1.0, help="multiplies every MIN_AREA_FRAC value")
    ap.add_argument("--single", nargs="*", default=SINGLE_INSTANCE, help="classes that can only appear once")
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--max-age-s", type=float, default=10.0, help="seconds a hidden item stays remembered by the tracker")
    ap.add_argument("--no-events", action="store_true", help="turn off motion detection and ADDED/REMOVED events")
    ap.add_argument("--show-motion", action="store_true", help="tint moving pixels red in the output video")
    ap.add_argument("--settle-s", type=float, default=0.8, help="item must stand still this long to count as ADDED")
    ap.add_argument("--gone-s", type=float, default=1.2, help="item must be undetected this long to count as REMOVED")
    ap.add_argument("--quiet-s", type=float, default=0.5, help="its spot must be motion-free this long before REMOVED")
    ap.add_argument("--motion-thr", type=float, default=0.06, help="fraction of a box that must change to count as moving")
    ap.add_argument("--initial-s", type=float, default=0.0, help="items settled within this many seconds of the start are logged INITIAL (already in cart)")
    ap.add_argument("--out-width", type=int, default=1080, help="output video width (input can stay 4K; smaller is faster)")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame (output keeps the same stride)")
    ap.add_argument("--debug-dir", default=None, help="save frames where a single-instance class had competing boxes")
    ap.add_argument("--max-debug", type=int, default=40)
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N processed frames (quick test)")
    ap.add_argument("--no-preview", action="store_true", help="do not open a live preview window")
    args = ap.parse_args()

    import openvino as ov

    xml = find_xml(args.model)
    names = load_names(Path(args.model))
    for s in args.single:
        if s not in names:
            raise SystemExit(f"--single class '{s}' is not in the model classes: {names}")

    core = ov.Core()
    print(f"Available devices: {core.available_devices}")
    compiled = core.compile_model(str(xml), args.device, {"PERFORMANCE_HINT": "LATENCY"})
    print(f"Model {xml.name} compiled for {args.device}")
    shp = compiled.input(0).shape
    size = (int(shp[2]), int(shp[3])) if len(shp) == 4 and shp[2] > 0 else (args.imgsz, args.imgsz)
    out_node = compiled.output(0)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    q = queue.Queue(maxsize=8)
    stop = threading.Event()
    reader = threading.Thread(target=frame_reader, args=(cap, q, args.stride, stop), daemon=True)
    reader.start()

    fps_eff = fps / args.stride
    tracker = Tracker(names, args.single, min_hits=args.min_hits, max_age=int(args.max_age_s * fps_eff))
    motion = engine = None
    if not args.no_events:
        from cart_events import MotionDetector, EventEngine
        motion = MotionDetector(fps_eff)
    writer, s = None, 1.0
    drop_stats = Counter()
    debug_saved, conflict_frames = 0, defaultdict(list)
    if args.debug_dir:
        Path(args.debug_dir).mkdir(parents=True, exist_ok=True)

    t0, n = time.time(), 0
    while True:
        item = q.get()
        if item is None:
            break
        idx, frame = item
        H, W = frame.shape[:2]  # use real (auto-rotated) frame size, not cap.get(WIDTH)
        if writer is None:
            s = min(1.0, args.out_width / W)
            ow, oh = int(W * s) // 2 * 2, int(H * s) // 2 * 2
            if motion is not None:
                engine = EventEngine(names, fps_eff, W, H, settle_s=args.settle_s, gone_s=args.gone_s,
                                     quiet_s=args.quiet_s, motion_thr=args.motion_thr,
                                     initial_s=args.initial_s, min_hits=args.min_hits)
            writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride, (ow, oh))
            print(f"Input {W}x{H}  ->  output {ow}x{oh}  ({total} frames)")

        blob, r, padx, pady = preprocess(frame, size)
        raw = compiled(blob)[out_node]
        dets = decode(raw, r, padx, pady, W, H, args.conf, args.iou)
        kept, dropped = clean_detections(dets, names, W * H, args)
        for d in dropped:
            drop_stats[(names[d["cls"]], d["reason"])] += 1

        # competing boxes for a single-instance class AFTER cleaning = a genuine conflict worth looking at
        conflict = [names[c] for c, k in Counter(d["cls"] for d in kept).items() if k > 1 and names[c] in args.single]
        for c in conflict:
            conflict_frames[c].append(idx)
        small = cv2.resize(frame, (int(W * s) // 2 * 2, int(H * s) // 2 * 2), interpolation=cv2.INTER_AREA)
        if args.debug_dir and (conflict or any(d["reason"] == "inside_same_class_box" and names[d["cls"]] in args.single for d in dropped)) \
                and debug_saved < args.max_debug:
            cv2.imwrite(str(Path(args.debug_dir) / f"frame_{idx:05d}.jpg"), draw_debug(small, kept, dropped, names, s))
            debug_saved += 1

        visible = tracker.update(kept, idx)
        if engine is not None:
            motion.update(frame)
            engine.update(tracker.tracks, motion, idx / fps, tracker)
            if args.show_motion:
                motion.overlay(small, s)
            out = draw_events_hud(draw_tracks(small, visible, names, s, engine), engine, motion, idx / fps)
        else:
            out = draw_hud(draw_tracks(small, visible, names, s), tracker, visible, names)
        writer.write(out)

        n += 1
        if args.max_frames and n >= args.max_frames:
            break
        if n % 25 == 0:
            el = time.time() - t0
            print(f"  frame {idx + 1}/{total}   {n / el:.1f} FPS", end="\r")
        if not args.no_preview:
            cv2.imshow("OpenVINO cart detection (q to quit)", cv2.resize(out, None, fx=0.5, fy=0.5) if out.shape[1] > 1000 else out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    stop.set()
    reader.join(timeout=5)
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    el = time.time() - t0

    # ---------------- summary ----------------
    print(f"\n\nDone: {n} frames in {el:.1f}s ({n / max(el, 1e-9):.1f} FPS).  Video: {args.output}")

    if engine is not None:
        events = engine.finish()
        base = Path(args.output).with_suffix("")
        Path(str(base) + ".events.json").write_text(json.dumps(
            {"events": events, "final_cart": dict(engine.cart())}, indent=2))
        keys = ["time_s", "event", "item", "track_id", "entered_from", "settled_at", "confidence", "left_toward"]
        with open(str(base) + ".events.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(events)
        print("\nEVENT LOG")
        for e in events:
            extra = e.get("entered_from") or e.get("left_toward") or ""
            print(f"  {e['time_s']:7.2f}s  {e['event']:8s} {e['item']:12s} {extra}")
        print("\nFINAL CART:", dict(engine.cart()) or "(empty)")
        for it in engine.items.values():
            if it.state == "MISSING":
                print(f"  [?] {names[it.cls]} was not visible at the end ({it.note or 'undecided'})")
        print(f"Saved: {base}.events.csv / .events.json")

    inventory = {}
    for c in range(len(names)):
        peak = tracker.peak.get(c, 0)
        if peak:
            uniq = sum(1 for k in tracker.ever_confirmed.values() if k == c)
            inventory[names[c]] = {"count": 1 if names[c] in args.single else peak,
                                   "peak_simultaneous": peak, "unique_tracks": uniq}
    print("\nCART INVENTORY (peak number of items seen at the same time):")
    for k, v in inventory.items():
        print(f"  {k:12s} x{v['count']}   (unique tracks seen: {v['unique_tracks']})")
    if drop_stats:
        print("\nBoxes removed by the cleaning rules (class, reason: count):")
        for (c, why), k in sorted(drop_stats.items(), key=lambda x: -x[1]):
            print(f"  {c:12s} {why:24s} {k}")
    for c, frames in conflict_frames.items():
        print(f"\n[!] '{c}' still had 2+ separate boxes on {len(frames)} frames "
              f"(e.g. frames {frames[:5]}). The tracker shows only one, but this usually means the model "
              f"confuses another object with '{c}'. Check --debug-dir frames and add them to your training set.")
    Path(args.output).with_suffix(".inventory.json").write_text(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
