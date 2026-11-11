# CrowdFlow: Drone Crowd-Safety Monitoring (MuJoCo prototype)

Two camera drones watch a 100-person concert floor, notice when the area in front of the stage is
becoming dangerously packed, and try to **prevent** a crush with escalating, non-physical
interventions: a red spotlight, a spoken announcement, and finally a notification to the security team.

This is a **simulation prototype for decision support**. It is not a validated crowd-safety system and
the drones never touch, push or herd anyone.

## Scenario: stage front

* Venue: 20 m × 15 m indoor floor, stage on the west wall behind a barrier, bar on the east wall.
  100 people average ~0.37 people/m², but the strip in front of the stage centre is only a few m².
* People: 60 fans, 28 casual attendees, 12 moving to and from the bar. Each fan decides **at a random
  time** to head for the centre of the stage barrier, and keeps pressing forward once there.
* Nothing is scripted to "surge" at a fixed time. Congestion emerges because many people want the
  same few square metres.

| Run (seed 2026, 90 s) | Peak density* | Time at ≥ 4 people/m² | Time at ≥ 5 people/m² |
|---|---|---|---|
| Drones observe only (`--no-interventions`) | 6.05 people/m² | 64 s | 59 s |
| Drones intervene | 4.46 people/m² | 6 s | 0 s |

\*Measured afterwards from simulator ground truth: most people inside any 1 m-radius circle. The drones
never see ground truth. Crowd-safety guidance commonly treats sustained densities above ~5 people/m² as
dangerous. The crowd model is stylised, so these numbers show the system works *in this simulation*,
not in a real venue.

## How it works (closed loop, every 0.1 s)

```text
 cameras ──► head detection ──► tracking ──► fleet crowd map + score C (1 Hz)
    ▲                                                     │
    │                                                     ▼
 people react (or not) ◄── spotlight / announcement ◄── fleet coordinator
                           security notification        (patrol, respond, escalate)
```

1. **Perception** (`perception/density_detector.py`): each drone's downward camera (640×480) is
   segmented for heads. In a dense crowd bodies merge into one blob but heads stay separate, so
   counting heads stays reliable. Heads are projected to floor coordinates using the drone's pose.
   *In the sim all heads share one colour. A real system would swap in a learned detector (e.g. YOLO);
   nothing downstream changes.*
2. **Tracking** (`perception/tracker.py`): detections are linked across frames (Hungarian matching,
   0.4 m gate), giving each person a temporary ID, a speed and a heading.
3. **Fleet crowd map** (`perception/zone_analytics.py`): both drones' tracks are fused into 2 m × 2 m
   overlapping zones. Each zone gets density, speed, growth, share stopped and flow dispersion, combined
   into a congestion score **C** (0–100) using the weights in the team brief. The status is
   NORMAL < 25 ≤ DEVELOPING < 50 ≤ HIGH < 75 ≤ CRITICAL. Unseen zones are *unknown*, never "empty".
   **Early warning:** each zone also gets a short-range prediction (+10/+20/+30 s), the higher of
   two signals: the density trend over the last 30 s, and the people currently *walking towards* the
   zone who would arrive in time. The second catches a wave of fans heading for the stage before the
   density itself rises.
4. **Fleet coordinator** (`brain/fleet_planner.py`): Alpha patrols the stage half and Bravo the bar
   half at 7 m. When a zone is DEVELOPING, or predicted to become dense, the nearer drone descends to
   5.5 m over it and the other climbs to 8 m to cover the gap, always keeping the stage front in view.
   It then runs the escalation ladder ("predicted" = density expected in 20 s):

| Level | Trigger | Action |
|---|---|---|
| 0 MONITOR | zone DEVELOPING for 3 s, or ≥ 1.5 people/m² and predicted ≥ 3.5 | responder moves over it to look closer |
| 1 SPOTLIGHT | HIGH for 2 s, DEVELOPING ≥ 2 people/m² and filling up, or predicted ≥ 3.5 | red spotlight marks the area |
| 2 ANNOUNCE | still HIGH 8 s later, CRITICAL, or predicted ≥ 4.0 | "please make space at the front", naming uncrowded side areas the drones **can see** are free (repeated after 10 s if not improving, up to 3 times) |
| 3 SECURITY | **only if density is not falling:** still HIGH 20 s after the announcement, or still CRITICAL 15 s after it | notify the security team (operator notification only) |
| RESOLVED | zone below HIGH for 8 s | spotlight off, back to patrol |

5. **People's response** (`sim/drone_env.py`): the spotlight and announcement only change what people
   know. Each person has a responsiveness (0.35–0.90) and a reaction delay (1–8 s). People in the
   spotlight may stop pushing forward. People who follow an announcement walk to the nearest named
   side area, and stop early if it is already busy. This is a stylised model, not a validated
   human-behaviour model.

**Lesson built into the design:** an early version sent everyone to one side area, which simply became
the next hotspot. The coordinator now spreads people over several observed, uncrowded areas.

### Adaptations of the brief's score for a standing audience
The brief's score was designed for walking flows. At a concert, standing still at 1–2 people/m² is
normal, so:
* speed, stopping and direction terms are scaled by a density gate `clamp((ρ − 2.0) / 1.5, 0, 1)`
* growth is scaled by `clamp(ρ − 1.0, 0, 1)`
* critical override B uses ρ ≥ 3.0 instead of 2.5.

All thresholds and weights are prototype choices, not validated safety limits.

## Run it

```bash
pip install -r requirements.txt
python config/generate_arena.py          # only needed after changing the venue
python run_simulation.py                  # 90 s, 30 FPS -> submission_video.mp4 + run_summary.json
python run_simulation.py --no-interventions --out baseline.mp4 --summary baseline.json
```

Options: `--duration`, `--fps` (multiple of 10), `--seed`, `--camera-fault` (corrupts Alpha's camera at
70–74 s to show the failsafe: spotlight off, climb to 8 m), `--no-video` (numbers only, faster; the
drones still use their cameras, so results are identical), `--quiet`.

### Evaluate across many crowds

```bash
python evaluate.py                          # seeds 1-10, drones on vs observe-only
python evaluate.py --seeds 10 --workers 4   # run 4 simulations in parallel
python evaluate.py --label before_prediction
```

Prints a comparison table and saves it to `results/<label>.json` (per-run details in `results/runs/`).

### Comparison video and plots

```bash
python side_by_side.py --seed 5                 # same crowd without / with CrowdFlow -> side_by_side.mp4
python side_by_side.py --seed 5 --clip 15 50 --gif   # plus a short clip and GIF for posts
python plot_results.py --label evaluation       # results/density_timeline.png, results/per_seed.png
```

The video shows the world view, the active drone's camera (tracked heads coloured moving / slow /
stopped), and an operator panel: the fleet crowd map, the escalation ladder, the worst zone's metrics,
and the operator log.

On a headless Linux machine, set `MUJOCO_GL=egl`.

## Repository

```text
config/generate_arena.py   venue + 100 pedestrians + 2 drones -> config/arena.xml (generated)
sim/drone_env.py           physics, crowd behaviour, response to drone outputs, ground-truth evaluation
perception/                head detector, tracker, fleet crowd map + score C
brain/fleet_planner.py     patrol, responder/overwatch roles, escalation ladder, failsafe
run_simulation.py          main loop, video + operator panel, evaluation summary
evaluate.py                multi-seed comparison: drones intervening vs observing only
side_by_side.py            comparison video (same crowd, without vs with drones) + clip/GIF
plot_results.py            timeline and per-crowd plots from evaluate.py results
run_all.py                 older standalone exit-blockage demo (separate, unchanged)
docs/                      pitch script and summary for the EARLIER 18-person version (out of date)
```

## Limitations

* Simulation only: simple cylinder pedestrians, stylised behaviour and compliance, indoor venue.
* Head detection relies on a uniform head colour in the sim.
* Latency figures printed at the end are from the machine you run on, not from drone hardware.
* No battery model yet (planned), no live operator controls yet (the video shows the panel only).
* The drone speaker's audibility over concert audio is assumed, not modelled.
