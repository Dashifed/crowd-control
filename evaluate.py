#!/usr/bin/env python3
"""
Multi-seed evaluation: runs the stage-front scenario for several crowds (seeds), each once
with the drones intervening and once with the drones only observing, then compares them.

Every number comes from simulator ground truth AFTER the run (peak density = most people in
any 1 m-radius circle). The drones never see ground truth.

Usage:
    python evaluate.py                     # seeds 1-10, one run at a time
    python evaluate.py --seeds 5 --workers 4
    python evaluate.py --label before_prediction   # saves results/before_prediction.json

Runs use --no-video, so 10 seeds x 2 conditions takes a few minutes on a laptop.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

import run_simulation


def one_run(job):
    seed, interventions, duration = job
    args = run_simulation.build_parser().parse_args([])
    args.seed = seed
    args.no_interventions = not interventions
    args.duration = duration
    args.no_video = True
    args.quiet = True
    args.summary = os.path.join("results", "runs", f"seed{seed}_{'on' if interventions else 'off'}.json")
    s = run_simulation.run(args)
    first_alert = next((e["t"] for e in s["events"] if e["text"].startswith("MONITOR")), None)
    return {"seed": seed, "interventions": interventions,
            "peak": s["gt_max_peak_density"],
            "s_ge4": s["gt_seconds_at_or_above_4"],
            "s_ge5": s["gt_seconds_at_or_above_5"],
            "security": s["security_alerted"],
            "first_alert_s": first_alert}


def summarise(rows, interventions):
    sel = [r for r in rows if r["interventions"] == interventions]
    peaks = np.array([r["peak"] for r in sel])
    ge5 = np.array([r["s_ge5"] for r in sel])
    out = {
        "runs": len(sel),
        "peak_mean": float(peaks.mean()), "peak_max": float(peaks.max()),
        "s_ge4_mean": float(np.mean([r["s_ge4"] for r in sel])),
        "s_ge5_mean": float(ge5.mean()),
        "runs_reaching_5": int((ge5 > 0).sum()),
    }
    if interventions:
        alerts = [r["first_alert_s"] for r in sel if r["first_alert_s"] is not None]
        out["security_alerts"] = int(sum(r["security"] for r in sel))
        out["first_alert_mean_s"] = float(np.mean(alerts)) if alerts else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10, help="evaluate seeds 1..N")
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--workers", type=int, default=1, help="parallel runs (try 4 on a laptop)")
    ap.add_argument("--label", default="evaluation", help="name of the results file")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)
    os.makedirs(os.path.join("results", "runs"), exist_ok=True)
    jobs = [(s, on, args.duration) for s in range(1, args.seeds + 1) for on in (False, True)]

    t0 = time.perf_counter()
    print(f"Running {len(jobs)} simulations ({args.seeds} seeds x drones on/off), "
          f"{args.workers} at a time ...")
    rows = []
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
            for r in ex.map(one_run, jobs):
                rows.append(r)
                print(f"  seed {r['seed']:2d} {'ON ' if r['interventions'] else 'OFF'}  "
                      f"peak {r['peak']:.2f}  >=5: {r['s_ge5']:3d}s")
    else:
        for job in jobs:
            r = one_run(job)
            rows.append(r)
            print(f"  seed {r['seed']:2d} {'ON ' if r['interventions'] else 'OFF'}  "
                  f"peak {r['peak']:.2f}  >=5: {r['s_ge5']:3d}s")

    off, on = summarise(rows, False), summarise(rows, True)
    result = {"label": args.label, "seeds": args.seeds, "duration_s": args.duration,
              "drones_observe_only": off, "drones_intervene": on, "runs": rows}
    path = os.path.join("results", f"{args.label}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)

    print(f"\nDone in {time.perf_counter() - t0:.0f}s. Saved {path}\n")
    print(f"| {args.seeds} seeds, {args.duration:.0f} s each | Drones observe only | Drones intervene |")
    print("|---|---|---|")
    print(f"| Mean peak density (people/m²) | {off['peak_mean']:.2f} | {on['peak_mean']:.2f} |")
    print(f"| Worst peak density | {off['peak_max']:.2f} | {on['peak_max']:.2f} |")
    print(f"| Mean seconds ≥ 4 people/m² | {off['s_ge4_mean']:.1f} | {on['s_ge4_mean']:.1f} |")
    print(f"| Mean seconds ≥ 5 people/m² | {off['s_ge5_mean']:.1f} | {on['s_ge5_mean']:.1f} |")
    print(f"| Runs that reached 5 people/m² | {off['runs_reaching_5']}/{off['runs']} "
          f"| {on['runs_reaching_5']}/{on['runs']} |")
    fa = on["first_alert_mean_s"]
    print(f"\nWith drones: first alert on average at {fa:.1f} s; "
          f"security notified in {on['security_alerts']}/{on['runs']} runs." if fa is not None else
          "\nWith drones: no alerts raised.")


if __name__ == "__main__":
    main()
