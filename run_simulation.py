#!/usr/bin/env python3
"""
CrowdFlow - stage-front scenario, 100 people, 2 drones (MuJoCo).

Closed loop, every 0.1 s:
    drone cameras -> head detection -> tracking -> (1 Hz) fleet crowd map + score C
    -> fleet coordinator (patrol / respond / escalation ladder) -> drone moves, spotlight,
    announcements -> people react (or not) -> cameras see the result.

Nothing is timed: congestion emerges as fans decide to move to the stage.
The drones only ever use camera-derived data. Ground truth is used once per second
for the evaluation summary at the end, never by the drones.

Usage:
    python run_simulation.py                    # 90 s run, 30 FPS video
    python run_simulation.py --fps 10           # faster rendering
    python run_simulation.py --no-interventions # baseline: drones watch but never act
    python run_simulation.py --camera-fault     # inject a camera fault at 70-74 s
    python run_simulation.py --no-video         # fast: numbers only, no rendering of the overview
"""

import argparse
import json
import os
import time
import numpy as np
import cv2

from sim.drone_env import CrowdFleetEnv, SPOTLIGHT_RADIUS, VENUE_X, VENUE_Y, BARRIER_X
from perception.density_detector import DensityPerceptionEngine
from perception.tracker import MultiTracker
from perception.zone_analytics import FleetCrowdMap, STATUS_NAMES
from brain.fleet_planner import FleetCoordinator, LADDER

FRAME_W, FRAME_H = 640, 480
PANEL_H = 230
CONTROL_HZ = 10
STATUS_BGR = {None: (90, 90, 90), 0: (80, 170, 80), 1: (40, 200, 230), 2: (30, 130, 255), 3: (40, 40, 230)}


# ---------------------------------------------------------------------- drawing
def put(img, text, org, scale=0.45, color=(235, 235, 240), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_world_pane(rgb, t, duration, coordinator, enabled, last_announcement):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if last_announcement is not None and t - last_announcement[0] < 6.0:
        cv2.rectangle(bgr, (0, FRAME_H - 34), (FRAME_W, FRAME_H), (30, 110, 200), -1)
        put(bgr, f"DRONE ANNOUNCEMENT: {last_announcement[1]}", (10, FRAME_H - 12), 0.45, thick=1)
    cv2.rectangle(bgr, (0, 0), (FRAME_W, 50), (18, 20, 26), -1)
    put(bgr, f"CROWDFLOW SIMULATION | t = {t:5.1f}s / {duration:.0f}s", (12, 20), 0.55, thick=2)
    mode = "INTERVENTIONS ON" if enabled else "BASELINE: INTERVENTIONS OFF"
    put(bgr, f"Alpha: {coordinator.state['alpha']}   Bravo: {coordinator.state['bravo']}   [{mode}]",
        (12, 42), 0.43, (170, 220, 255))
    return bgr


def draw_drone_pane(perc, tracks, drone, pos, state, detector, crowd_map, t, beacon):
    img = perc["bgr"].copy()
    for tr in tracks:
        u, v = detector.world_to_pixel(tr.pos, pos)
        col = (60, 220, 60) if tr.speed > 0.3 else (0, 200, 255) if tr.slow_updates < 3 else (60, 60, 255)
        cv2.circle(img, (u, v), 6, col, 2)
    worst = crowd_map.worst_zone(t, fresh_only=False)
    if worst is not None and worst.status >= 1:
        x0, x1, y0, y1 = worst.rect
        p0 = detector.world_to_pixel((x0, y1), pos)
        p1 = detector.world_to_pixel((x1, y0), pos)
        cv2.rectangle(img, p0, p1, STATUS_BGR[worst.status], 2)
        put(img, f"{STATUS_NAMES[worst.status]} C={worst.score:.0f}", (p0[0], max(14, p0[1] - 6)),
            0.45, STATUS_BGR[worst.status], 2)
    cv2.rectangle(img, (0, 0), (FRAME_W, 50), (18, 20, 26), -1)
    put(img, f"DRONE {drone.upper()} CAMERA | z={pos[2]:.1f}m | {state}", (12, 20), 0.55, thick=2)
    put(img, f"heads detected: {len(perc['detections'])}  tracks: {len(tracks)}  "
             f"cam health: {perc['vision_confidence']:.2f}  spotlight: {'ON' if beacon else 'off'}",
        (12, 42), 0.42, (190, 200, 210))
    put(img, "circles: green moving / amber slow / red stopped", (12, FRAME_H - 10), 0.4, (200, 200, 200))
    return img


def draw_panel(crowd_map, coordinator, t, drone_pos, beacon):
    panel = np.full((PANEL_H, FRAME_W * 2, 3), (22, 24, 30), np.uint8)

    # --- fleet crowd map (left) ---
    s = 13.0
    ox, oy = 14, 18
    to_px = lambda x, y: (int(ox + (x + VENUE_X) * s), int(oy + (VENUE_Y - y) * s))
    put(panel, "FLEET CROWD MAP (camera-derived)", (ox, 13), 0.42, (200, 200, 210))
    cv2.rectangle(panel, to_px(-VENUE_X, VENUE_Y), to_px(VENUE_X, -VENUE_Y), (70, 70, 80), 1)
    cv2.rectangle(panel, to_px(-VENUE_X, 5.5), to_px(BARRIER_X, -5.5), (90, 60, 80), -1)
    put(panel, "STAGE", (to_px(-VENUE_X, 0)[0] + 1, to_px(0, 0)[1]), 0.3, (220, 200, 230))
    for z in crowd_map.zones:                       # 1 m cells coloured by the zone centred there
        fresh = crowd_map.freshness(z, t)
        col = STATUS_BGR[z.status] if fresh != "unknown" else STATUS_BGR[None]
        if fresh == "stale":
            col = tuple(int(c * 0.55) for c in col)
        p0, p1 = to_px(z.c[0] - 0.5, z.c[1] + 0.5), to_px(z.c[0] + 0.5, z.c[1] - 0.5)
        cv2.rectangle(panel, p0, (p1[0] - 1, p1[1] - 1), col, -1)
    for d, colr in (("alpha", (40, 200, 245)), ("bravo", (245, 165, 40))):
        p = to_px(*drone_pos[d][:2])
        if beacon[d]:
            cv2.circle(panel, p, int(SPOTLIGHT_RADIUS * s), (40, 40, 255), 1)
        cv2.circle(panel, p, 5, colr, -1)
        put(panel, d[0].upper(), (p[0] + 6, p[1] - 4), 0.4, colr, 1)
    legend_y = oy + int(2 * VENUE_Y * s) + 14
    for k, (name, col) in enumerate([("normal", 0), ("developing", 1), ("high", 2), ("critical", 3),
                                     ("unknown", None)]):
        x = ox + k * 54
        cv2.rectangle(panel, (x, legend_y - 8), (x + 8, legend_y), STATUS_BGR[col], -1)
        put(panel, name, (x + 11, legend_y), 0.3, (200, 200, 200))
    put(panel, f"coverage {crowd_map.coverage(t) * 100:.0f}%", (ox + 200, 13), 0.38, (170, 170, 180))

    # --- escalation ladder + metrics (middle) ---
    mx = 300
    put(panel, "ESCALATION LADDER", (mx, 18), 0.45, (200, 200, 210))
    inc = coordinator.incident
    level = inc["level"] if inc else -1
    for k, name in enumerate(LADDER):
        y = 34 + k * 30
        active = k <= level
        col = (40, 40, 230) if (k == 3 and active) else (30, 160, 255) if active else (70, 72, 80)
        cv2.rectangle(panel, (mx, y), (mx + 160, y + 24), col, -1 if active else 1)
        put(panel, f"{k}  {name}", (mx + 8, y + 17), 0.45, (255, 255, 255) if active else (150, 150, 160))
    z = inc.get("zone") if inc else crowd_map.worst_zone(t)
    if z is not None and z.status is not None:
        fmt = lambda v, f: "n/a" if v is None else f.format(v)
        lines = [f"zone ({z.c[0]:+.0f},{z.c[1]:+.0f}) {STATUS_NAMES[z.status]}  C={z.score:.0f}",
                 f"density {z.rho:.2f} p/m2   growth {fmt(z.growth, '{:+.2f}')} /min",
                 f"speed {fmt(z.speed, '{:.2f}')} m/s   stopped {fmt(z.p_stop, '{:.0%}')}",
                 f"trend +10/+20/+30s: " + ("n/a" if z.pred is None else
                     " / ".join(f"{z.pred[h]:.1f}" for h in (10, 20, 30)) + " p/m2"),
                 f"data: {crowd_map.freshness(z, t)}"]
        for k, line in enumerate(lines):
            put(panel, line, (mx + 175, 44 + k * 22), 0.4, (210, 210, 220))
    if inc is None:
        put(panel, "no active incident", (mx + 175, 44), 0.42, (130, 200, 130))

    # --- security banner + event log (right) ---
    ex = 760
    if coordinator.security_alert:
        cv2.rectangle(panel, (ex, 6), (FRAME_W * 2 - 10, 34), (30, 30, 200), -1)
        put(panel, "SECURITY TEAM NOTIFIED - assistance requested at stage front", (ex + 8, 26), 0.45,
            (255, 255, 255), 1)
    put(panel, "OPERATOR LOG", (ex, 54), 0.45, (200, 200, 210))
    for k, (et, text) in enumerate(coordinator.events[-7:]):
        put(panel, f"{et:5.1f}s {text[:84]}", (ex, 74 + k * 21), 0.33, (210, 210, 220))
    return panel


# ---------------------------------------------------------------------- main loop
def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--fps", type=int, default=30, help="video FPS (multiple of 10)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-interventions", action="store_true", help="baseline: observe only")
    ap.add_argument("--camera-fault", action="store_true", help="inject Alpha camera fault 70-74 s")
    ap.add_argument("--out", default="submission_video.mp4")
    ap.add_argument("--summary", default="run_summary.json")
    ap.add_argument("--no-video", action="store_true",
                    help="skip the overview camera and video file (much faster; drones still use their cameras)")
    ap.add_argument("--quiet", action="store_true", help="only print the final evaluation")
    return ap


def run(args):
    """Runs one simulation and returns the evaluation summary (also written to args.summary)."""
    assert args.fps % CONTROL_HZ == 0, "--fps must be a multiple of 10"
    say = (lambda *a, **k: None) if args.quiet else print

    base = os.path.dirname(os.path.abspath(__file__))
    env = CrowdFleetEnv(frame_width=FRAME_W, frame_height=FRAME_H, seed=args.seed)
    detector = DensityPerceptionEngine(FRAME_W, FRAME_H)
    trackers = {"alpha": MultiTracker(), "bravo": MultiTracker()}
    crowd_map = FleetCrowdMap()
    coordinator = FleetCoordinator(interventions_enabled=not args.no_interventions)

    video_path = os.path.join(base, args.out)
    writer = None if args.no_video else cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (FRAME_W * 2, FRAME_H + PANEL_H))
    frames_per_tick = args.fps // CONTROL_HZ
    steps_per_tick = int(round(1.0 / CONTROL_HZ / env.sim_dt))
    dt = 1.0 / CONTROL_HZ
    n_ticks = int(args.duration * CONTROL_HZ)

    lat = {"perception": [], "tracking": [], "analytics": [], "brain": []}
    timeline = []
    last_announcement = None
    wall0 = time.perf_counter()
    say(f"Running {args.duration:.0f}s, {env.num_pedestrians} people, "
          f"interventions {'OFF' if args.no_interventions else 'ON'} ...")

    for tick in range(n_ticks):
        t = env.data.time

        # 1. Perception (each drone's own camera)
        percepts, tracks, cam_ok = {}, {}, {}
        tp = 0.0
        for d in ("alpha", "bravo"):
            rgb = env.render_camera(f"cam_{d}")
            fault = args.camera_fault and d == "alpha" and 70.0 <= t <= 74.0
            percepts[d] = detector.analyze_frame(rgb, coordinator.pos[d], simulate_dropout=fault)
            tp += percepts[d]["latency_ms"]
            cam_ok[d] = percepts[d]["vision_confidence"] >= 0.5
            t0 = time.perf_counter()
            tracks[d] = trackers[d].update(percepts[d]["detections"], t, percepts[d]["footprint"]) \
                if cam_ok[d] else []
            lat["tracking"].append((time.perf_counter() - t0) * 1000.0)
        lat["perception"].append(tp)

        # 2. Fleet crowd map at 1 Hz
        if tick % CONTROL_HZ == 0:
            t0 = time.perf_counter()
            crowd_map.update(t, [{"drone": d, "xy": coordinator.pos[d][:2],
                                  "footprint": percepts[d]["footprint"], "tracks": tracks[d]}
                                 for d in ("alpha", "bravo") if cam_ok[d]])
            lat["analytics"].append((time.perf_counter() - t0) * 1000.0)

        # 3. Decide
        n_events = len(coordinator.events)
        out = coordinator.update(crowd_map, cam_ok, t, dt)
        lat["brain"].append(out["brain_ms"])
        for et, text in coordinator.events[n_events:]:
            say(f"  [{et:5.1f}s] {text}")

        # 4. Act
        for d in ("alpha", "bravo"):
            env.set_beacon(d, coordinator.beacon[d])
        ann = coordinator.pending_announcement
        if ann is not None:
            n = env.announce(ann["drone"], ann["hotspot"], ann["relief"])
            last_announcement = (t, "please make space at the front - move to the sides"
                                 if len(ann["relief"]) else "please stop pushing forward")
            coordinator.log(t, f"{n} people chose to follow the announcement")
            say(f"  [{t:5.1f}s] {coordinator.events[-1][1]}")

        # 5. Physics + video frames
        if writer is None:
            env.step(coordinator.pos, substeps=steps_per_tick)
        else:
            write_frames(env, writer, args, coordinator, crowd_map, detector, percepts, tracks,
                         t, last_announcement, frames_per_tick, steps_per_tick)

        # Offline evaluation sample (ground truth, NOT used by the drones)
        if tick % CONTROL_HZ == 0:
            peak, where = env.gt_peak_local_density(1.0)
            timeline.append({"t": round(t, 1), "gt_peak_density": round(float(peak), 2),
                             "incident_level": coordinator.incident["level"] if coordinator.incident else -1})
            if tick % (10 * CONTROL_HZ) == 0:
                say(f"t={t:5.1f}s  true peak density {peak:4.2f} p/m2  "
                    f"| alpha {coordinator.state['alpha']}, bravo {coordinator.state['bravo']}")

    if writer is not None:
        writer.release()
    env.close()
    return finish(args, base, coordinator, timeline, lat, wall0, video_path, say)


def write_frames(env, writer, args, coordinator, crowd_map, detector, percepts, tracks,
                 t, last_announcement, frames_per_tick, steps_per_tick):
    """Advances physics for one control tick while writing the video frames."""
    sub = steps_per_tick // frames_per_tick
    show = coordinator.incident["responder"] if coordinator.incident else "alpha"
    drone_img = draw_drone_pane(percepts[show], tracks[show], show, coordinator.pos[show],
                                coordinator.state[show], detector, crowd_map, t,
                                coordinator.beacon[show])
    panel = draw_panel(crowd_map, coordinator, t, coordinator.pos, coordinator.beacon)
    for f in range(frames_per_tick):
        env.step(coordinator.pos, substeps=sub)
        world = draw_world_pane(env.render_camera("world_cam"), env.data.time, args.duration,
                                coordinator, not args.no_interventions, last_announcement)
        writer.write(np.vstack([np.hstack([world, drone_img]), panel]))
    leftover = steps_per_tick - sub * frames_per_tick
    if leftover:
        env.step(coordinator.pos, substeps=leftover)


def finish(args, base, coordinator, timeline, lat, wall0, video_path, say):
    peaks = np.array([s["gt_peak_density"] for s in timeline])
    summary = {
        "interventions": not args.no_interventions,
        "seed": args.seed,
        "duration_s": args.duration,
        "gt_max_peak_density": float(peaks.max()),
        "gt_seconds_at_or_above_4": int((peaks >= 4.0).sum()),
        "gt_seconds_at_or_above_5": int((peaks >= 5.0).sum()),
        "security_alerted": any("SECURITY" in e[1] for e in coordinator.events),
        "events": [{"t": round(et, 1), "text": tx} for et, tx in coordinator.events],
        "latency_ms_mean": {k: round(float(np.mean(v)), 3) for k, v in lat.items() if v},
        "timeline": timeline,
    }
    with open(os.path.join(base, args.summary), "w") as f:
        json.dump(summary, f, indent=1)

    say("\n" + "=" * 70)
    say(f"Done in {time.perf_counter() - wall0:.0f}s wall time."
        + ("" if args.no_video else f" Video: {video_path}"))
    say("EVALUATION (simulator ground truth, peak density in any 1 m-radius circle):")
    say(f"  max peak density      {summary['gt_max_peak_density']:.2f} people/m2")
    say(f"  seconds >= 4 p/m2     {summary['gt_seconds_at_or_above_4']}")
    say(f"  seconds >= 5 p/m2     {summary['gt_seconds_at_or_above_5']}")
    say("Mean stage latency on THIS machine (ms): " +
        ", ".join(f"{k} {v}" for k, v in summary["latency_ms_mean"].items()))
    say(f"Summary written to {args.summary}")
    return summary


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
