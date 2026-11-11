#!/usr/bin/env python3
"""
Side-by-side comparison video: the same crowd (same seed) without and with CrowdFlow.

    left   drones only watch     (--no-interventions)
    right  drones intervene
    bottom live chart of the true peak density in both runs, with the escalation steps marked

Usage:
    python side_by_side.py                         # seed 5, 90 s -> side_by_side.mp4
    python side_by_side.py --seed 5 --fps 20
    python side_by_side.py --clip 15 50 --gif      # also export a short clip (+ GIF) for posts

Densities in the chart are simulator ground truth, measured for evaluation only.
"""

import argparse
import json
import os
import numpy as np
import cv2

import run_simulation

W, H = 640, 480              # one world pane (cropped from each run's video)
LABEL_H, CHART_H = 46, 240
OUT_W, OUT_H = 2 * W, LABEL_H + H + CHART_H

# Colours (BGR). Validated pair from the reference palette, dark-surface steps.
SURFACE = (25, 26, 26)       # #1a1a19
TEXT = (236, 239, 240)       # #f0efec
TEXT_2 = (183, 194, 195)     # #c3c2b7
GRID = (70, 72, 74)
OFF_COL = (38, 89, 217)      # #d95926 orange  - drones only watch
ON_COL = (229, 135, 57)      # #3987e5 blue    - drones intervene


def put(img, text, org, scale=0.5, color=TEXT, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def simulate(seed, interventions, duration, fps, base):
    args = run_simulation.build_parser().parse_args([])
    tag = "on" if interventions else "off"
    args.seed, args.duration, args.fps = seed, duration, fps
    args.no_interventions = not interventions
    args.out = os.path.join("results", f"sbs_seed{seed}_{tag}.mp4")
    args.summary = os.path.join("results", f"sbs_seed{seed}_{tag}.json")
    args.quiet = True
    print(f"  simulating seed {seed}, drones {'intervene' if interventions else 'only watch'} ...")
    run_simulation.run(args)
    with open(os.path.join(base, args.summary)) as f:
        return os.path.join(base, args.out), json.load(f)


def label_strip(off, on):
    img = np.full((LABEL_H, OUT_W, 3), SURFACE, np.uint8)
    for x0, col, title, sub in ((0, OFF_COL, "WITHOUT CROWDFLOW", "drones only watch"),
                                (W, ON_COL, "WITH CROWDFLOW", "drones spotlight, announce, alert security")):
        cv2.rectangle(img, (x0 + 14, 14), (x0 + 30, 30), col, -1)
        put(img, title, (x0 + 40, 28), 0.62, TEXT, 2)
        put(img, sub, (x0 + 40 + (230 if "WITHOUT" in title else 200), 28), 0.45, TEXT_2)
    return img


class DensityChart:
    """Live line chart of both runs' true peak density, revealed as the video plays."""

    def __init__(self, off, on, duration):
        self.t = np.array([s["t"] for s in off["timeline"]])
        self.off = np.array([s["gt_peak_density"] for s in off["timeline"]])
        self.on = np.array([s["gt_peak_density"] for s in on["timeline"]])
        self.duration = duration
        self.events = []
        for e in on["events"]:
            for key, short in (("SPOTLIGHT", "spotlight"), ("ANNOUNCE", "announce"),
                               ("SECURITY", "security")):
                if e["text"].startswith(key) and not e["text"].startswith("ANNOUNCE #"):
                    self.events.append((e["t"], short))
        self.x0, self.x1 = 70, OUT_W - 150
        self.y0, self.y1 = 30, CHART_H - 34
        self.ymax = 7.0

    def px(self, t, v):
        x = self.x0 + (self.x1 - self.x0) * t / self.duration
        y = self.y1 - (self.y1 - self.y0) * min(v, self.ymax) / self.ymax
        return int(round(x)), int(round(y))

    def draw(self, now):
        img = np.full((CHART_H, OUT_W, 3), SURFACE, np.uint8)
        put(img, "Peak crowd density (people per m2, simulator ground truth)", (self.x0, 20), 0.48, TEXT)
        for v in range(0, 8):                                  # recessive grid
            p = self.px(0, v)
            cv2.line(img, (self.x0, p[1]), (self.x1, p[1]), GRID if v not in (4, 5) else TEXT_2, 1)
            put(img, f"{v}", (self.x0 - 22, p[1] + 5), 0.4, TEXT_2)
        put(img, "4  movement restricted", (self.x1 + 8, self.px(0, 4)[1] + 5), 0.4, TEXT_2)
        put(img, "5  dangerous", (self.x1 + 8, self.px(0, 5)[1] + 5), 0.4, TEXT_2)
        for s in range(0, int(self.duration) + 1, 10):
            p = self.px(s, 0)
            put(img, f"{s}s", (p[0] - 10, self.y1 + 20), 0.4, TEXT_2)

        for et, name in self.events:                          # escalation steps (ON run)
            if et <= now:
                p = self.px(et, self.ymax)
                cv2.line(img, (p[0], self.y0), (p[0], self.y1), ON_COL, 1)
                put(img, name, (p[0] + 3, self.y0 + 12 + 14 * [e[1] for e in self.events].index(name) % 42),
                    0.38, TEXT_2)

        k = int(np.searchsorted(self.t, now, side="right"))
        for series, col in ((self.off, OFF_COL), (self.on, ON_COL)):
            if k >= 2:
                pts = np.array([self.px(tt, vv) for tt, vv in zip(self.t[:k], series[:k])], np.int32)
                cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)
            if k >= 1:
                end = self.px(self.t[k - 1], series[k - 1])
                cv2.circle(img, end, 5, col, -1, cv2.LINE_AA)
                cv2.circle(img, end, 7, SURFACE, 2, cv2.LINE_AA)     # surface ring
        if k >= 1:                                                     # current values, text ink
            put(img, f"without: {self.off[k - 1]:.1f}", (self.x1 + 8, self.y0 + 10), 0.45, TEXT)
            put(img, f"with:    {self.on[k - 1]:.1f}", (self.x1 + 8, self.y0 + 30), 0.45, TEXT)
            cv2.rectangle(img, (self.x1 + 118, self.y0 + 1), (self.x1 + 128, self.y0 + 11), OFF_COL, -1)
            cv2.rectangle(img, (self.x1 + 118, self.y0 + 21), (self.x1 + 128, self.y0 + 31), ON_COL, -1)
        return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="side_by_side.mp4")
    ap.add_argument("--clip", type=float, nargs=2, metavar=("START", "END"),
                    help="also write a short clip between these times (seconds)")
    ap.add_argument("--gif", action="store_true", help="also write the clip as a GIF (needs Pillow)")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)
    os.makedirs("results", exist_ok=True)
    print(f"Side-by-side for seed {args.seed}:")
    off_path, off = simulate(args.seed, False, args.duration, args.fps, base)
    on_path, on = simulate(args.seed, True, args.duration, args.fps, base)

    cap_off, cap_on = cv2.VideoCapture(off_path), cv2.VideoCapture(on_path)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (OUT_W, OUT_H))
    clip_writer, gif_frames = None, []
    if args.clip:
        clip_name = os.path.splitext(args.out)[0] + "_clip.mp4"
        clip_writer = cv2.VideoWriter(clip_name, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (OUT_W, OUT_H))

    labels = label_strip(off, on)
    chart = DensityChart(off, on, args.duration)
    frame_i = 0
    while True:
        ok1, f_off = cap_off.read()
        ok2, f_on = cap_on.read()
        if not (ok1 and ok2):
            break
        now = frame_i / args.fps
        frame = np.vstack([labels, np.hstack([f_off[:H, :W], f_on[:H, :W]]), chart.draw(now)])
        writer.write(frame)
        if args.clip and args.clip[0] <= now <= args.clip[1]:
            clip_writer.write(frame)
            if args.gif and frame_i % max(1, args.fps // 6) == 0:       # GIF at ~6 fps, small
                gif_frames.append(cv2.cvtColor(cv2.resize(frame, (OUT_W * 3 // 8, OUT_H * 3 // 8),
                                                          interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        frame_i += 1
    writer.release()
    print(f"Wrote {args.out}")
    if clip_writer is not None:
        clip_writer.release()
        print(f"Wrote {clip_name}")
    if gif_frames:
        try:
            from PIL import Image
            imgs = [Image.fromarray(f) for f in gif_frames]
            gif_name = os.path.splitext(args.out)[0] + "_clip.gif"
            imgs = [im.quantize(colors=96) for im in imgs]
            imgs[0].save(gif_name, save_all=True, append_images=imgs[1:],
                         duration=int(1000 / 6), loop=0, optimize=True)
            print(f"Wrote {gif_name}")
        except ImportError:
            print("Pillow not installed - skipped the GIF (pip install pillow)")

    print(f"\nSeed {args.seed}: peak density {off['gt_max_peak_density']:.2f} without vs "
          f"{on['gt_max_peak_density']:.2f} with CrowdFlow; seconds >= 5 people/m2: "
          f"{off['gt_seconds_at_or_above_5']} vs {on['gt_seconds_at_or_above_5']}")


if __name__ == "__main__":
    main()
