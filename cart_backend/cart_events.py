"""
Motion detection + "item ADDED to / REMOVED from the cart" events.

How it decides (all timings are in SECONDS, so they do not depend on the video fps):

  ADDED    a confirmed item has been detected AND is standing still (no motion inside its box)
           for `settle_s`.  An item that is only waved in front of the camera and pulled back
           never settles, so it is never counted.

  REMOVED  an item that was in the cart stops being detected for `gone_s`, AND
             - there was motion on its spot around the time it vanished (a hand took it), AND
             - that spot is quiet again (the hand has left, so the item would be visible if it
               were still there), AND
             - it is not hidden underneath another item.
           A hand that merely passes over an item for a moment does not remove it, because the
           item reappears and the state returns to IN_CART.

Motion = frame differencing on a small grayscale image, after cancelling hand-held camera shake
with phase correlation (without this, a shaking camera looks like "everything moves").
"""
from collections import deque, Counter
import cv2
import numpy as np


# ----------------------------------------------------------------------------------------
class MotionDetector:
    def __init__(self, fps, width=256, diff_thr=22, lag_s=0.1, cam_motion_frac=0.45):
        self.w, self.diff_thr, self.cam_frac = width, diff_thr, cam_motion_frac
        self.lag = max(1, round(lag_s * fps))
        self.hist = deque(maxlen=self.lag + 1)
        self.mask = None
        self.scale = None          # small px per original px
        self.global_ratio = 0.0
        self.camera_moving = False
        self._k3 = np.ones((3, 3), np.uint8)
        self._k5 = np.ones((5, 5), np.uint8)

    def update(self, frame):
        H, W = frame.shape[:2]
        self.scale = self.w / W
        h = int(round(H * self.scale))
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (self.w, h), interpolation=cv2.INTER_AREA),
                                          cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if len(self.hist) < self.lag:
            self.hist.append(g)
            self.mask = np.zeros_like(g, np.uint8)
            self.global_ratio, self.camera_moving = 0.0, False
            return self.mask
        prev = self.hist[0]
        self.hist.append(g)
        try:  # cancel camera shake: shift the old frame onto the new one
            (dx, dy), _ = cv2.phaseCorrelate(np.float32(prev), np.float32(g))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            prev = cv2.warpAffine(prev, M, (g.shape[1], g.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        except cv2.error:
            pass
        m = (cv2.absdiff(prev, g) > self.diff_thr).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._k3)
        self.mask = cv2.dilate(m, self._k5)
        self.global_ratio = float(self.mask.mean())
        self.camera_moving = self.global_ratio > self.cam_frac
        return self.mask

    def box_ratio(self, box):
        """Fraction of the box area (original-frame coordinates) that is moving."""
        if self.mask is None or self.scale is None:
            return 0.0
        x1, y1, x2, y2 = [int(round(v * self.scale)) for v in box]
        h, w = self.mask.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return float(self.mask[y1:y2, x1:x2].mean())

    def overlay(self, img, s_out):
        """Tint moving pixels red on the output image (for --show-motion)."""
        if self.mask is None:
            return img
        m = cv2.resize(self.mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        img[m] = (0.55 * img[m] + 0.45 * np.array((0, 0, 255))).astype(np.uint8)
        return img


# ----------------------------------------------------------------------------------------
def _edge(box, W, H):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    d = {"left": cx / W, "right": 1 - cx / W, "top": cy / H, "bottom": 1 - cy / H}
    return min(d, key=d.get)


def _cover(a, b):
    """fraction of box a that lies inside box b"""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / (max(1e-9, (a[2] - a[0]) * (a[3] - a[1])))


class Item:
    def __init__(self, tr, t, W, H):
        self.id, self.cls = tr.id, tr.cls
        self.state = "PENDING"            # PENDING -> IN_CART <-> MISSING -> REMOVED
        self.first_t = t
        self.entry_edge = _edge(tr.box, W, H)
        self.last_seen_t = t
        self.stable_since = None
        self.lost_t = None
        self.quiet_since = None
        self.box = tr.box.copy()
        self.motion = deque()             # (t, ratio inside box)
        self.note = ""


class EventEngine:
    def __init__(self, names, fps, W, H, settle_s=0.8, gone_s=1.2, quiet_s=0.5, motion_thr=0.06,
                 evidence_s=1.5, cover_thr=0.7, give_up_s=5.0, initial_s=0.0, min_hits=3):
        self.names, self.fps, self.W, self.H = names, fps, W, H
        self.settle_s, self.gone_s, self.quiet_s = settle_s, gone_s, quiet_s
        self.motion_thr, self.evidence_s, self.cover_thr = motion_thr, evidence_s, cover_thr
        self.give_up_s, self.initial_s, self.min_hits = give_up_s, initial_s, min_hits
        self.items = {}
        self.events = []                  # list of dicts
        self.popups = deque()             # (until_t, text, color) for on-screen banners

    # -------------------------------------------------------------------------
    def cart(self):
        return Counter(self.names[i.cls] for i in self.items.values() if i.state in ("IN_CART", "MISSING"))

    def status(self, track_id):
        it = self.items.get(track_id)
        return it.state if it else None

    def _emit(self, kind, it, t, now=None, **extra):
        ev = {"time_s": round(t, 2), "event": kind, "item": self.names[it.cls], "track_id": it.id, **extra}
        self.events.append(ev)
        self.popups.append(((now if now is not None else t) + 2.5, f"{kind}: {self.names[it.cls]}", (0, 200, 0) if kind != "REMOVED" else (0, 0, 230)))
        print(f"\n[{t:6.2f}s] {kind:8s} {self.names[it.cls]:12s} (#{it.id}) {extra if extra else ''}")

    # -------------------------------------------------------------------------
    def update(self, tracks, motion, t, tracker):
        if motion.camera_moving:           # whole picture moves: do not judge anything this frame
            return
        confirmed = [tr for tr in tracks if tr.hits >= self.min_hits]
        ids = {tr.id for tr in confirmed}

        for tr in confirmed:
            it = self.items.get(tr.id)
            if it is None:
                it = self.items[tr.id] = Item(tr, t, self.W, self.H)
            it.box = tr.box.copy()
            visible = tr.seen_now
            if visible:
                it.last_seen_t = t
            m = motion.box_ratio(it.box)
            it.motion.append((t, m))
            while it.motion and it.motion[0][0] < t - 4.0:
                it.motion.popleft()
            self._step(it, visible, m, t, tracker)

        # tracks that vanished from the tracker without an event (flicker) are forgotten
        for tid in list(self.items):
            if tid not in ids and self.items[tid].state == "PENDING":
                del self.items[tid]

    # -------------------------------------------------------------------------
    def _step(self, it, visible, m, t, tracker):
        still = m < self.motion_thr

        if it.state == "PENDING":
            if visible and still:
                it.stable_since = it.stable_since if it.stable_since is not None else t
                if t - it.stable_since >= self.settle_s:
                    it.state = "IN_CART"
                    kind = "INITIAL" if it.first_t <= self.initial_s else "ADDED"
                    self._emit(kind, it, it.first_t, now=t, entered_from=it.entry_edge, settled_at=round(t, 2))
            else:
                it.stable_since = None
                if not visible and t - it.last_seen_t > self.gone_s:      # only touched / waved, never placed
                    del self.items[it.id]
                    tracker.drop(it.id)

        elif it.state == "IN_CART":
            if not visible:
                it.state, it.lost_t, it.quiet_since = "MISSING", it.last_seen_t, None

        elif it.state == "MISSING":
            if visible:                                                    # was only hidden (hand / shadow)
                it.state, it.lost_t = "IN_CART", None
                return
            if still:
                it.quiet_since = it.quiet_since if it.quiet_since is not None else t
            else:
                it.quiet_since = None
            missing_for = t - it.lost_t
            if missing_for < self.gone_s or it.quiet_since is None or t - it.quiet_since < self.quiet_s:
                return

            # Hidden under another item only if that item showed up around the moment this one vanished.
            # (An item lying ON TOP of a long-standing box, e.g. a pen on the chess box, is not "hidden".)
            hidden_under = any(o.id != it.id and o.state in ("IN_CART", "PENDING") and o.cls != it.cls
                               and o.first_t >= it.lost_t - self.evidence_s
                               and _cover(it.box, o.box) >= self.cover_thr for o in self.items.values())
            if hidden_under:
                it.note = "hidden under another item"
                return

            peak = max((mm for tt, mm in it.motion if it.lost_t - self.evidence_s <= tt <= t), default=0.0)
            if peak >= self.motion_thr:
                conf = "high"
            elif missing_for >= self.give_up_s:
                conf = "low (no motion seen)"
            else:
                return
            it.state, it.note = "REMOVED", ""
            self._emit("REMOVED", it, it.lost_t, now=t, confidence=conf, left_toward=_edge(it.box, self.W, self.H))
            tracker.drop(it.id)

    # -------------------------------------------------------------------------
    def finish(self):
        """items still MISSING at the end of the video are reported as unresolved"""
        for it in self.items.values():
            if it.state == "MISSING":
                it.note = it.note or "not visible at end of video"
        return self.events

    def popup_lines(self, t):
        while self.popups and self.popups[0][0] < t:
            self.popups.popleft()
        return list(self.popups)
