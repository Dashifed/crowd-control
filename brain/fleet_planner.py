"""
Fleet coordinator: decides what Alpha and Bravo do, from the fleet crowd map only
(camera-derived). It never reads simulator ground truth.

Drone roles
    PATROL      Alpha watches the stage half, Bravo the bar half, both sweeping north/south
                at 7 m so the whole floor is revisited.
    RESPONDER   the drone nearest an incident descends to 5.5 m over it.
    OVERWATCH   the other drone climbs to 8 m over mid-floor to cover the gap.
    FAILSAFE    camera unhealthy -> spotlight off, climb to 8 m and hold.

Escalation ladder for one incident (all thresholds are prototype choices):
    0 MONITOR    worst zone DEVELOPING for 3 s -> responder moves over it to look closer
    1 SPOTLIGHT  HIGH for 2 s of fresh data, or DEVELOPING at >= 2 people/m2 and filling
                 up (growth >= 0.3 people/m2/min) for 4 s -> red spotlight marks the area
    2 ANNOUNCE   still HIGH 10 s after the spotlight, or CRITICAL -> announcement asking
                 people to move to the less crowded side areas (chosen from the map)
    3 SECURITY   still HIGH 15 s after the announcement, or still CRITICAL 10 s after it ->
                 notify security / bodyguards (operator notification only)
    RESOLVED     zone below HIGH for 6 s -> spotlight off, drones return to patrol
"""

import time
import numpy as np

PATROL_Z, RESPOND_Z, OVERWATCH_Z, FAILSAFE_Z = 7.0, 5.5, 8.0, 8.0
MAX_SPEED = 2.0                     # m/s
SWEEP_AMPLITUDE, SWEEP_PERIOD = 2.8, 24.0
PATROL_X = {"alpha": -4.5, "bravo": 4.5}
OVERWATCH_XY = np.array([1.0, 0.0])

STATUS = ["NORMAL", "DEVELOPING", "HIGH", "CRITICAL"]
LADDER = ["MONITOR", "SPOTLIGHT", "ANNOUNCE", "SECURITY"]
# Candidate side/back areas people can be asked to move to (stage side, then further back).
RELIEF_CANDIDATES = np.array([[-7.4, 4.5], [-7.4, -4.5], [-6.4, 5.8], [-6.4, -5.8],
                              [-5.4, 3.8], [-5.4, -3.8], [-4.4, 5.5], [-4.4, -5.5],
                              [-3.4, 2.5], [-3.4, -2.5]])
TARGET_RELIEF_DENSITY = 1.5          # people/m2 we are willing to fill a relief area up to
MAX_ANNOUNCEMENTS = 3


class FleetCoordinator:
    def __init__(self, interventions_enabled: bool = True):
        self.pos = {"alpha": np.array([PATROL_X["alpha"], 0.0, PATROL_Z]),
                    "bravo": np.array([PATROL_X["bravo"], 0.0, PATROL_Z])}
        self.state = {"alpha": "PATROL", "bravo": "PATROL"}
        self.beacon = {"alpha": False, "bravo": False}
        self.enabled = interventions_enabled
        self.incident = None            # dict describing the active incident
        self.developing_since = None
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
                target = np.array([OVERWATCH_XY[0], OVERWATCH_XY[1], OVERWATCH_Z])
            else:
                self.state[d] = "PATROL"
                self.beacon[d] = False
                target = self._patrol_target(d, t)
            self._move_towards(d, target, dt)

        return {"brain_ms": (time.perf_counter() - t0) * 1000.0}

    def _update_incident(self, crowd_map, worst, t):
        inc = self.incident
        if inc is None:
            if worst is not None and worst.status >= 1:
                self.developing_since = self.developing_since or t
                if t - self.developing_since >= 3.0:
                    responder = min(("alpha", "bravo"),
                                    key=lambda d: np.linalg.norm(self.pos[d][:2] - worst.centroid))
                    self.incident = {"xy": worst.centroid.copy(), "level": 0, "level_since": t,
                                     "responder": responder, "start": t, "calm_since": None,
                                     "critical_since": None, "confirm_since": None,
                                     "peak_status": worst.status}
                    self.log(t, f"MONITOR: {STATUS[worst.status]} "
                                f"crowding near ({worst.centroid[0]:+.1f}, {worst.centroid[1]:+.1f}); "
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

        def escalate(new_level, reason):
            inc["level"], inc["level_since"] = new_level, t
            self.log(t, f"{LADDER[new_level]}: {reason}")

        if level == 0:
            fresh = crowd_map.freshness(z, t) == "fresh"
            filling = status == 1 and z.rho >= 2.0 and z.growth is not None and z.growth >= 0.3
            confirm = fresh and (status >= 2 or filling)
            inc["confirm_since"] = (inc["confirm_since"] or t) if confirm else None
            if inc["confirm_since"] is not None and t - inc["confirm_since"] >= (2.0 if status >= 2 else 4.0):
                why = "HIGH confirmed" if status >= 2 else f"area filling up ({z.growth:+.1f} p/m2/min)"
                escalate(1, f"{why} (C={z.score:.0f}, {z.rho:.1f} p/m2) - red spotlight marks the area")
        elif level == 1 and ((status >= 2 and t - since >= 8.0) or critical):
            self._announce(crowd_map, inc, t, escalate, critical)
        elif level >= 2:
            if level == 2 and ((status >= 2 and t - since >= 15.0) or
                               (inc["critical_since"] is not None and t - inc["critical_since"] >= 5.0
                                and t - since >= 10.0)):
                escalate(3, f"still {'CRITICAL' if critical else 'HIGH'} after announcement "
                            "- SECURITY TEAM ALERTED (notification)")
                self.security_alert = True
            if (status >= 2 and t - inc["last_announce"] >= 15.0
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

    def _announce(self, crowd_map, inc, t, escalate, critical, repeat=False):
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
            escalate(2, ("CRITICAL - " if critical else "still HIGH - ") + msg)
        self.pending_announcement = {"drone": inc["responder"], "hotspot": inc["xy"].copy(),
                                     "relief": relief}
