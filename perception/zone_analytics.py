"""
Fleet crowd map: fuses both drones' tracks into per-zone crowd measurements and a
congestion score C (0-100), following the team brief (Agent Handoff v10.1, section 3-4)
with the adaptations listed under ADAPTATIONS below.

Zones: overlapping 2 m x 2 m windows on a 1 m lattice over the walkable floor, so a
cluster is never split across a zone boundary.

Per zone, from camera-derived tracks only:
    density  rho      people / visible walkable m2
    speed ratio rv    median track speed / 1.2 m/s
    growth   g_rho    6 * (rho_now - rho_10s_ago), people/m2/min
    stopped  P_stop   share of tracks below 0.2 m/s for >= 3 consecutive updates
    dispersion D_dir  1 - |mean unit heading| (needs >= 5 moving tracks)
    duration          seconds since the zone qualified as an incident

    C = 100 * (0.25 C_rho + 0.20 C_v + 0.20 C_g + 0.15 C_s + 0.10 C_d + 0.10 C_p)
Unavailable components are dropped and the remaining weights renormalised.

Prediction, two signals, the higher one wins:
  * trend (brief section 3.4): least squares on the zone's smoothed density over the last
    30 s (needs >= 3 samples spanning >= 10 s), projected +10/+20/+30 s ahead;
  * inflow: tracked people within 8 m who are walking towards the zone and would reach it
    within the horizon, added to the people already there. A wave of fans heading for the
    stage shows up here seconds before the density itself rises.
Both are short-horizon estimates, not forecasts of how a crowd will behave.

Freshness: <= 2 s fresh, <= 5 s stale, older = unknown. Unknown is NOT zero.

ADAPTATIONS (deliberate, for a standing concert audience rather than walking flows):
  * Speed, stopping and direction components are scaled by a density gate
    clamp((rho - 2.0) / 1.5, 0, 1). An audience standing still at 1-2 people/m2 is
    watching the show, not congested; standing still at 3.5 people/m2 is a warning sign.
  * Growth is scaled by clamp(rho - 1.0, 0, 1): one person walking into an empty
    zone is a large relative increase but not a congestion signal.
  * Critical override B uses rho >= 3.0 (brief: 2.5). Front rows standing still while
    the area fills up is normal at 2.5 people/m2 at a concert.
  * Analytics run at 1 Hz, zones are 2 m x 2 m instead of 5 m x 5 m (smaller venue).
These are prototype engineering choices, not validated safety thresholds.
"""

import numpy as np

STAGE_BARRIER_X = -8.4
WALKABLE = (STAGE_BARRIER_X, 9.0, -7.5, 7.5)     # xmin, xmax, ymin, ymax (bar at x > 9)
WIN = 2.0
V_REF = 1.2
FRESH_S, STALE_S = 2.0, 5.0
STATUS_NAMES = ["NORMAL", "DEVELOPING", "HIGH", "CRITICAL"]
PRED_WINDOW_S = 30.0
PRED_HORIZONS = (10, 20, 30)
WEIGHTS = {"rho": 0.25, "v": 0.20, "g": 0.20, "s": 0.15, "d": 0.10, "p": 0.10}


def clamp01(x):
    return float(min(1.0, max(0.0, x)))


def rect_overlap(a, b):
    w = min(a[1], b[1]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[2], b[2])
    return max(0.0, w) * max(0.0, h)


def status_of(score):
    return 0 if score < 25 else 1 if score < 50 else 2 if score < 75 else 3


class Zone:
    def __init__(self, cx, cy):
        self.c = np.array([cx, cy])
        self.rect = (cx - WIN / 2, cx + WIN / 2, cy - WIN / 2, cy + WIN / 2)
        self.walk_area = rect_overlap(self.rect, WALKABLE)
        self.rho = self.speed = self.disp = None
        self.p_stop = None
        self.growth = None
        self.score = None
        self.status = None
        self.last_seen = -np.inf
        self.history = []                 # (t, rho_smoothed), last 30 s
        self.pred = None                  # {10: rho, 20: rho, 30: rho} or None
        self.inbound = 0                  # people currently heading into this zone
        self.change5 = None               # density change over the last ~5 s
        self.pre_since = None             # time C_pre first went >= 25
        self.override_since = None
        self.critical_override = False
        self.n_tracks = 0
        self.centroid = self.c.copy()


class FleetCrowdMap:
    def __init__(self):
        xs = np.arange(STAGE_BARRIER_X + WIN / 2, WALKABLE[1] - WIN / 2 + 1e-6, 1.0)
        ys = np.arange(-7.5 + WIN / 2, 7.5 - WIN / 2 + 1e-6, 1.0)
        self.zones = [Zone(x, y) for x in xs for y in ys]
        self.xs, self.ys = xs, ys

    # ------------------------------------------------------------------ update
    def update(self, t, observations):
        """
        observations: list of dicts, one per drone with a healthy camera:
            {"drone": name, "xy": drone xy, "footprint": rect, "tracks": [Track, ...]}
        """
        for obs in observations:            # stop counters are per track, once per update
            for tr in obs["tracks"]:
                tr.slow_updates = tr.slow_updates + 1 if tr.speed < 0.2 else 0

        movers = self._movers(observations)
        for z in self.zones:
            best = None
            for obs in observations:
                vis = rect_overlap(rect_overlap_rect(z.rect, WALKABLE), obs["footprint"])
                frac = vis / max(z.walk_area, 1e-9)
                if frac >= 0.7 and z.walk_area >= 1.5:
                    key = (round(frac, 1), -np.linalg.norm(z.c - obs["xy"]))
                    if best is None or key > best[0]:
                        best = (key, obs, vis)
            if best is not None:
                self._measure(z, t, best[1], best[2], movers)
            self._refresh_status(z, t)

    @staticmethod
    def _movers(observations, min_speed=0.4):
        """Moving tracks from all drones, de-duplicated where both drones see the same person."""
        out = []
        for obs in observations:
            for tr in obs["tracks"]:
                if tr.speed >= min_speed and all(np.linalg.norm(tr.pos - p) > 0.3 for p, _ in out):
                    out.append((tr.pos.copy(), tr.vel.copy()))
        return out

    def _measure(self, z, t, obs, vis_area, movers=()):
        x0, x1, y0, y1 = z.rect
        inside = [tr for tr in obs["tracks"] if x0 <= tr.pos[0] < x1 and y0 <= tr.pos[1] < y1]
        z.n_tracks = len(inside)
        z.centroid = np.mean([tr.pos for tr in inside], axis=0) if inside else z.c.copy()
        ema = lambda new, old: new if old is None else 0.4 * new + 0.6 * old

        z.rho = ema(len(inside) / vis_area, z.rho)
        if len(inside) >= 3:
            z.speed = ema(float(np.median([tr.speed for tr in inside])), z.speed)
            z.p_stop = float(np.mean([tr.slow_updates >= 3 for tr in inside]))
        else:
            z.speed, z.p_stop = None, None
        moving = [tr.vel / tr.speed for tr in inside if tr.speed > 0.3]
        z.disp = ema(1.0 - float(np.linalg.norm(np.mean(moving, axis=0))), z.disp) if len(moving) >= 5 else None

        z.history.append((t, z.rho))
        z.history = [(ht, hr) for ht, hr in z.history if t - ht <= PRED_WINDOW_S]
        past = [hr for ht, hr in z.history if 9.0 <= t - ht <= 11.0]
        z.growth = 6.0 * (z.rho - past[0]) if past else None
        recent = [hr for ht, hr in z.history if 4.0 <= t - ht <= 6.0]
        z.change5 = z.rho - recent[0] if recent else None
        z.pred = self._predict(z, t)

        # Inflow: people walking towards this zone who would arrive within each horizon.
        x0, x1, y0, y1 = z.rect
        etas = []
        for p, v in movers:
            if x0 <= p[0] < x1 and y0 <= p[1] < y1:
                continue
            to_z = z.c - p
            dist = float(np.linalg.norm(to_z))
            if dist > 8.0:
                continue
            speed = float(np.linalg.norm(v))
            if np.dot(to_z / dist, v / speed) > 0.9:          # heading at the zone
                etas.append(max(0.0, dist - WIN / 2) / speed)
        z.inbound = sum(1 for e in etas if e <= 10.0)
        flow = {h: (len(inside) + sum(1 for e in etas if e <= h)) / vis_area for h in PRED_HORIZONS}
        z.pred = flow if z.pred is None else {h: max(z.pred[h], flow[h]) for h in PRED_HORIZONS}
        z.last_seen = t
        self._score(z, t)

    @staticmethod
    def _predict(z, t):
        """Linear trend of smoothed density over the last 30 s, projected ahead."""
        if len(z.history) < 3:
            return None
        ts = np.array([ht - t for ht, _ in z.history])
        rs = np.array([hr for _, hr in z.history])
        if ts.max() - ts.min() < 10.0:
            return None
        slope, intercept = np.polyfit(ts, rs, 1)
        return {h: float(max(0.0, intercept + slope * h)) for h in PRED_HORIZONS}

    def _score(self, z, t):
        gate = clamp01((z.rho - 2.0) / 1.5)
        comp = {"rho": clamp01((z.rho - 1.0) / 2.5)}
        rv = None
        if z.speed is not None:
            rv = z.speed / V_REF
            comp["v"] = clamp01((0.75 - rv) / 0.55) * gate
        if z.growth is not None:
            comp["g"] = clamp01((z.growth - 0.10) / 0.50) * clamp01(z.rho - 1.0)
        if z.p_stop is not None:
            comp["s"] = clamp01((z.p_stop - 0.10) / 0.50) * gate
        if z.disp is not None:
            comp["d"] = z.disp * gate

        def weighted(c):
            w = sum(WEIGHTS[k] for k in c)
            return 100.0 * sum(WEIGHTS[k] * v for k, v in c.items()) / w

        c_pre = weighted(comp)                       # duration forced to zero
        if c_pre >= 25:
            z.pre_since = t if z.pre_since is None else z.pre_since
        else:
            z.pre_since = None
        if z.pre_since is not None and t - z.pre_since >= 5.0:   # qualified incident
            comp["p"] = clamp01((t - z.pre_since - 5.0) / 25.0)
        z.score = weighted(comp)

        # Critical override (must hold for 3 s)
        cond_a = z.rho >= 3.5 and rv is not None and rv <= 0.30
        cond_b = (z.rho >= 3.0 and z.p_stop is not None and z.p_stop >= 0.5
                  and z.growth is not None and z.growth >= 0.30)
        if cond_a or cond_b:
            z.override_since = t if z.override_since is None else z.override_since
        else:
            z.override_since = None
        z.critical_override = z.override_since is not None and t - z.override_since >= 3.0

    def _refresh_status(self, z, t):
        age = t - z.last_seen
        if age > STALE_S or z.score is None:
            z.status = None                          # unknown
        else:
            z.status = 3 if z.critical_override else status_of(z.score)

    # ------------------------------------------------------------------ queries
    def freshness(self, z, t):
        age = t - z.last_seen
        return "fresh" if age <= FRESH_S else "stale" if age <= STALE_S else "unknown"

    def worst_zone(self, t, fresh_only=True):
        cands = [z for z in self.zones if z.status is not None
                 and (not fresh_only or self.freshness(z, t) == "fresh")]
        if not cands:
            return None
        return max(cands, key=lambda z: (z.status, z.score))

    def zone_at(self, xy):
        return min(self.zones, key=lambda z: np.linalg.norm(z.c - np.asarray(xy)))

    def coverage(self, t):
        known = sum(1 for z in self.zones if self.freshness(z, t) != "unknown")
        return known / len(self.zones)


def rect_overlap_rect(a, b):
    return (max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3]))
