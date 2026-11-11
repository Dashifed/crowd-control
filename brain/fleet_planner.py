"""
Fleet coordinator: decides what Alpha and Bravo do, from the fleet crowd map only
(camera-derived). It never reads simulator ground truth.

Drone roles
    PATROL      Alpha watches the stage half, Bravo the bar half, both sweeping north/south
                at 7 m so the whole floor is revisited.
    RESPONDER   the drone nearest an incident descends to 5.5 m over it.
    OVERWATCH   the other drone climbs to 8 m to cover the gap. If the responder has left
                the stage front (the venue's priority area), overwatch covers the stage front.
    FAILSAFE    camera unhealthy -> spotlight off, climb to 8 m and hold.

Escalation ladder for one incident (all thresholds are prototype choices).
"pred" = the zone's density trend projected 20 s ahead (perception/zone_analytics.py).
    0 MONITOR    worst zone DEVELOPING for 3 s, or a zone already at >= 1.5 people/m2 heading
                 for >= 3.5 (pred) for 2 s -> responder moves over it to look closer
    1 SPOTLIGHT  HIGH for 2 s of fresh data, DEVELOPING at >= 2 people/m2 and filling up
                 for 4 s, or pred >= 3.5 for 2 s -> red spotlight marks the area
    2 ANNOUNCE   still HIGH 8 s after the spotlight, CRITICAL, or pred >= 4.0 after 3 s of
                 spotlight -> announcement asking people to move to uncrowded side areas
    3 SECURITY   only if the area is NOT improving (density not falling over the last 5 s):
                 still HIGH 20 s after the announcement, or still CRITICAL 15 s after it
                 (people need time to react and walk) -> notify security (notification only)
    RESOLVED     zone below HIGH for 8 s -> spotlight off, drones return to patrol
"""

import time
import numpy as np

PATROL_Z, RESPOND_Z, OVERWATCH_Z, FAILSAFE_Z = 7.0, 5.5, 8.0, 8.0
MAX_SPEED = 2.0                     # m/s
SWEEP_AMPLITUDE, SWEEP_PERIOD = 2.8, 24.0
PATROL_X = {"alpha": -4.5, "bravo": 4.5}
OVERWATCH_XY = np.array([3.4, 0.0])           # covers mid-floor and the bar side
STAGE_WATCH_XY = np.array([-4.5, 0.0])         # covers the stage front (priority area)
STAGE_FRONT_X = -6.0                           # x below this = stage front

STATUS = ["NORMAL", "DEVELOPING", "HIGH", "CRITICAL"]
LADDER = ["MONITOR", "SPOTLIGHT", "ANNOUNCE", "SECURITY"]
# Candidate side/back areas people can be asked to move to (stage side, then further back).
RELIEF_CANDIDATES = np.array([[-7.4, 4.5], [-7.4, -4.5], [-6.4, 5.8], [-6.4, -5.8],
                              [-5.4, 3.8], [-5.4, -3.8], [-4.4, 5.5], [-4.4, -5.5],
                              [-3.4, 2.5], [-3.4, -2.5]])
TARGET_RELIEF_DENSITY = 1.5          # people/m2 we are willing to fill a relief area up to
MAX_ANNOUNCEMENTS = 3
PRED_H = 20                          # seconds ahead used by the ladder
PRED_WATCH, PRED_SPOT, PRED_ANNOUNCE = 3.5, 3.5, 4.0     # people/m2


def pred_of(z):
    return None if z is None or z.pred is None else z.pred[PRED_H]


class FleetCoordinator:
    def __init__(self, interventions_enabled: bool = True):
        self.pos = {"alpha": np.array([PATROL_X["alpha"], 0.0, PATROL_Z]),
                    "bravo": np.array([PATROL_X["bravo"], 0.0, PATROL_Z])}
        self.state = {"alpha": "PATROL", "bravo": "PATROL"}
        self.beacon = {"alpha": False, "bravo": False}
        self.enabled = interventions_enabled
        self.incident = None            # dict describing the active incident
        self.developing_since = None
        self.heading_since = None       # a zone's trend has predicted high density since then
        self.events = []                # (time, text) operator log
        self.pending_announcement = None
        self.security_alert = False

    # ------------------------------------------------------------------ helpers
    def log(self, t, text):
        self.events.append((t, text))

    def _patrol_target(self, d, t):
        phase = 2 * np.pi * t / SWEEP_PERIOD + (0.0 if d == "alpha" else np.pi)
        return np.array([PATROL_X[d], SWEEP_AMPLITUDE * np.sin(phase), PATROL_Z])

    def _move_towards(self, d, target, dt):
        step = target - self.pos[d]
        dist = np.linalg.norm(step)
        if dist > MAX_SPEED * dt:
            step *= MAX_SPEED * dt / dist
        self.pos[d] = self.pos[d] + step

    def _choose_relief(self, crowd_map, t, hotspot_xy):
        """
        Picks observed, uncrowded areas to send people to (never an unknown area: unseen is
        not the same as empty). Spreads people over several areas, at most two per side,
        so the relief areas do not become the next hotspot.
        """
        options = []
        for p in RELIEF_CANDIDATES:
            z = crowd_map.zone_at(p)
            if (z.status == 0 and crowd_map.freshness(z, t) != "unknown"
                    and z.rho < TARGET_RELIEF_DENSITY - 0.5
                    and np.linalg.norm(p - hotspot_xy) > 2.5):
                capacity = (TARGET_RELIEF_DENSITY - z.rho) * 4.0       # people, per 2x2 m zone
                options.append((-capacity, p))
        options.sort(key=lambda o: o[0])
        picked = []
        for _, p in options:
            if sum(np.sign(q[1]) == np.sign(p[1]) for q in picked) < 2:
                picked.append(p)
            if len(picked) == 4:
                break
        return picked

    # ------------------------------------------------------------------ main update
    def update(self, crowd_map, camera_ok: dict, t: float, dt: float):
        t0 = time.perf_counter()
        self.pending_announcement = None
        worst = crowd_map.worst_zone(t)

        if self.enabled:
            self._update_incident(crowd_map, worst, t)

        # --- motion targets per role ---
        responder = self.incident["responder"] if self.incident else None
        for d in ("alpha", "bravo"):
            if not camera_ok[d]:
                if self.state[d] != "FAILSAFE":
                    self.log(t, f"{d.upper()}: camera unhealthy -> failsafe climb, spotlight off")
                self.state[d] = "FAILSAFE"
                self.beacon[d] = False
                target = np.array([self.pos[d][0], self.pos[d][1], FAILSAFE_Z])
            elif d == responder:
                self.state[d] = "RESPONDER"
                c = self.incident["xy"]
                target = np.array([c[0], c[1], RESPOND_Z])
                self.beacon[d] = self.incident["level"] >= 1
            elif responder is not None:
                self.state[d] = "OVERWATCH"
                self.beacon[d] = False
                xy = OVERWATCH_XY if self.incident["xy"][0] < STAGE_FRONT_X else STAGE_WATCH_XY
                target = np.array([xy[0], xy[1], OVERWATCH_Z])
            else:
                self.state[d] = "PATROL"
                self.beacon[d] = False
                target = self._patrol_target(d, t)
            self._move_towards(d, target, dt)

        return {"brain_ms": (time.perf_counter() - t0) * 1000.0}

    def _update_incident(self, crowd_map, worst, t):
        inc = self.incident
        if inc is None:
            # Early warning: a zone whose trend says it will be dense in 20 s.
            heading = [z for z in crowd_map.zones if crowd_map.freshness(z, t) == "fresh"
                       and z.rho is not None and z.rho >= 1.5 and (pred_of(z) or 0) >= PRED_WATCH]
            if heading:
                self.heading_since = self.heading_since or t
            else:
                self.heading_since = None
            early = heading and t - self.heading_since >= 2.0
            if early and (worst is None or worst.status < 1 or
                          (pred_of(max(heading, key=pred_of)) or 0) > (worst.rho or 0)):
                worst = max(heading, key=pred_of)
            if worst is not None and (worst.status >= 1 or early):
                self.developing_since = self.developing_since or t
                if t - self.developing_since >= 3.0 or early:
                    responder = min(("alpha", "bravo"),
                                    key=lambda d: np.linalg.norm(self.pos[d][:2] - worst.centroid))
                    self.incident = {"xy": worst.centroid.copy(), "level": 0, "level_since": t,
                                     "responder": responder, "start": t, "calm_since": None,
                                     "critical_since": None, "confirm_since": None,
                                     "peak_status": worst.status}
                    why = (f"density heading for {pred_of(worst):.1f} p/m2 in {PRED_H}s" if early
                           else f"{STATUS[worst.status]} crowding")
                    self.log(t, f"MONITOR: {why} near ({worst.centroid[0]:+.1f}, {worst.centroid[1]:+.1f}); "
                                f"{responder.upper()} moving in to inspect")
            else:
                self.developing_since = None
            return

        # Track the incident: follow the worst zone near the incident location.
        z = crowd_map.zone_at(inc["xy"])
        near = [zz for zz in crowd_map.zones if zz.status is not None
                and np.linalg.norm(zz.c - inc["xy"]) <= 2.5 and crowd_map.freshness(zz, t) == "fresh"]
        if near:
            z = max(near, key=lambda zz: (zz.status, zz.score))
            inc["xy"] = 0.8 * inc["xy"] + 0.2 * z.centroid
        status = z.status if z.status is not None else -1
        inc["peak_status"] = max(inc["peak_status"], status)
        inc["zone"] = z

        critical = status == 3
        inc["critical_since"] = (inc["critical_since"] or t) if critical else None
        level, since = inc["level"], inc["level_since"]
        fresh = crowd_map.freshness(z, t) == "fresh"
        pred = pred_of(z) if fresh else None
        improving = z.change5 is not None and z.change5 < -0.15     # density falling over ~5 s

        def escalate(new_level, reason):
            inc["level"], inc["level_since"] = new_level, t
            self.log(t, f"{LADDER[new_level]}: {reason}")

        if level == 0:
            filling = status == 1 and z.rho >= 2.0 and z.growth is not None and z.growth >= 0.3
            heading = pred is not None and pred >= PRED_SPOT
            confirm = fresh and (status >= 2 or filling or heading)
            inc["confirm_since"] = (inc["confirm_since"] or t) if confirm else None
            hold = 2.0 if (status >= 2 or heading) else 4.0
            if inc["confirm_since"] is not None and t - inc["confirm_since"] >= hold:
                why = ("HIGH confirmed" if status >= 2 else
                       f"heading for {pred:.1f} p/m2 in {PRED_H}s" if heading else
                       f"area filling up ({z.growth:+.1f} p/m2/min)")
                escalate(1, f"{why} (C={z.score:.0f}, {z.rho:.1f} p/m2) - red spotlight marks the area")
        elif level == 1 and ((status >= 2 and t - since >= 8.0) or critical or
                             (pred is not None and pred >= PRED_ANNOUNCE and t - since >= 3.0)):
            self._announce(crowd_map, inc, t, escalate, critical,
                           heading=None if (status >= 2 and t - since >= 8.0) or critical else pred)
        elif level >= 2:
            if level == 2 and not improving and (
                    (status >= 2 and t - since >= 20.0) or
                    (inc["critical_since"] is not None and t - inc["critical_since"] >= 5.0
                     and t - since >= 15.0)):
                escalate(3, f"still {'CRITICAL' if critical else 'HIGH'} and not improving after "
                            "the announcement - SECURITY TEAM ALERTED (notification)")
                self.security_alert = True
            repeat_after = 15.0 if improving else 10.0
            if (status >= 2 and t - inc["last_announce"] >= repeat_after
                    and inc["announcements"] < MAX_ANNOUNCEMENTS):
                self._announce(crowd_map, inc, t, escalate, critical, repeat=True)

        # Resolution: an inspected-but-unconfirmed area must calm to NORMAL; an area we
        # intervened on must stay below HIGH for 8 s.
        calm = status == 0 if level == 0 else 0 <= status <= 1
        if calm:
            inc["calm_since"] = inc["calm_since"] or t
            if t - inc["calm_since"] >= (6.0 if level == 0 else 8.0):
                self.log(t, f"RESOLVED: area back to {STATUS[status]} after {t - inc['start']:.0f}s"
                            + (" - spotlight off" if level >= 1 else "") + ", resuming patrol")
                self.incident = None
                self.developing_since = None
                self.security_alert = False
        else:
            inc["calm_since"] = None

    def _announce(self, crowd_map, inc, t, escalate, critical, repeat=False, heading=None):
        relief = self._choose_relief(crowd_map, t, inc["xy"])
        inc["announcements"] = inc.get("announcements", 0) + 1
        inc["last_announce"] = t
        if relief:
            where = ", ".join(f"({p[0]:+.0f},{p[1]:+.0f})" for p in relief)
            msg = f"'please make space at the front' -> side areas {where}"
        else:
            msg = "'please stop pushing forward' (no uncrowded area observed)"
        if repeat:
            self.log(t, f"ANNOUNCE #{inc['announcements']}: {msg}")
        else:
            lead = ("CRITICAL - " if critical else
                    f"heading for {heading:.1f} p/m2 - " if heading is not None else "still HIGH - ")
            escalate(2, lead + msg)
        self.pending_announcement = {"drone": inc["responder"], "hotspot": inc["xy"].copy(),
                                     "relief": relief}
