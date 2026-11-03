"""Test 4: tracker identifiers are isolated per Tracker instance, even across threads."""
import threading

import numpy as np

from cart_detector import Tracker


def test_tracker_id_isolation():
    names = ["book", "calculator", "chess_board", "chess_box", "watch"]
    single = ["calculator", "chess_board", "chess_box", "book"]
    results = {}

    def run(label):
        tracker = Tracker(names, single)
        tracker.update([{"cls": 0, "score": 0.9, "box": np.array([10.0, 10.0, 100.0, 100.0])}], 0)
        tracker.update([{"cls": 1, "score": 0.9, "box": np.array([200.0, 200.0, 300.0, 300.0])}], 1)
        results[label] = [t.id for t in tracker.tracks]

    threads = [threading.Thread(target=run, args=(n,)) for n in ("t1", "t2")]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert results["t1"] == [1, 2]
    assert results["t2"] == [1, 2]


def test_single_instance_class_keeps_one_slot():
    """Two 'book' detections in one frame must still produce a single book track."""
    tracker = Tracker(["book", "calculator"], ["book", "calculator"])
    dets = [
        {"cls": 0, "score": 0.9, "box": np.array([10.0, 10.0, 100.0, 100.0])},
        {"cls": 0, "score": 0.6, "box": np.array([300.0, 300.0, 400.0, 400.0])},
    ]
    tracker.update(dets, 0)
    assert len([t for t in tracker.tracks if t.cls == 0]) == 1
