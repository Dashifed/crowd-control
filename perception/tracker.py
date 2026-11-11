"""
Multi-object tracker: links head detections across frames so each person keeps an ID.

Tracking is what turns "how many people are here" into "how are they moving": speed,
whether they are stopped, and which direction they are heading all need a track.

Method (deliberately simple, ByteTrack-style in spirit):
  * predict each track forward with its current velocity,
  * match predictions to detections with the Hungarian algorithm inside a 0.4 m gate,
  * a new track is confirmed after 3 hits; a track is dropped after 5 missed frames.
Velocity is a finite difference over ~0.5 s of history, which suppresses pixel jitter.
"""

from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment

GATE_M = 0.40
CONFIRM_HITS = 3
MAX_MISSES = 5
VEL_WINDOW = 5          # frames (0.5 s at 10 Hz)


class Track:
    _next_id = 0

    def __init__(self, xy, t):
        self.id = Track._next_id
        Track._next_id += 1
        self.pos = np.asarray(xy, dtype=float)
        self.vel = np.zeros(2)
        self.hits, self.misses = 1, 0
        self.history = deque([(t, self.pos.copy())], maxlen=VEL_WINDOW + 1)
        self.slow_updates = 0          # consecutive analytics updates below stop speed

    @property
    def confirmed(self):
        return self.hits >= CONFIRM_HITS

    @property
    def speed(self):
        return float(np.linalg.norm(self.vel))

    def update(self, xy, t):
        self.pos = np.asarray(xy, dtype=float)
        self.history.append((t, self.pos.copy()))
        t0, p0 = self.history[0]
        if t - t0 > 1e-6 and len(self.history) >= 3:
            self.vel = (self.pos - p0) / (t - t0)
        self.hits += 1
        self.misses = 0


class MultiTracker:
    def __init__(self):
        self.tracks = []
        self.last_t = None

    def update(self, detections, t, footprint=None):
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        dets = np.asarray(detections).reshape(-1, 2)

        if self.tracks and len(dets):
            pred = np.array([tr.pos + tr.vel * dt for tr in self.tracks])
            cost = np.linalg.norm(pred[:, None, :] - dets[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(cost)
            matched_t, matched_d = set(), set()
            for r, c in zip(rows, cols):
                if cost[r, c] <= GATE_M:
                    self.tracks[r].update(dets[c], t)
                    matched_t.add(r)
                    matched_d.add(c)
        else:
            matched_t, matched_d = set(), set()

        for i, tr in enumerate(self.tracks):
            if i not in matched_t:
                tr.misses += 1
        for j in range(len(dets)):
            if j not in matched_d:
                self.tracks.append(Track(dets[j], t))

        def keep(tr):
            if tr.misses > MAX_MISSES:
                return False
            if footprint is not None and tr.misses > 0:      # walked out of view
                x0, x1, y0, y1 = footprint
                if not (x0 <= tr.pos[0] <= x1 and y0 <= tr.pos[1] <= y1):
                    return False
            return True

        self.tracks = [tr for tr in self.tracks if keep(tr)]
        return [tr for tr in self.tracks if tr.confirmed and tr.misses == 0]
