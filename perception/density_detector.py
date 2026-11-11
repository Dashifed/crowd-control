"""
Onboard perception: downward RGB frame -> head detections -> ground-plane positions.

Detector: colour segmentation of heads. In the simulator every head is the same yellow,
which keeps the detector simple and fast. Bodies merge into one blob in a dense crowd;
heads do not (two people's heads are always >= ~0.14 m apart), so counting heads
stays reliable at high density. A real deployment would replace this class with a learned
detector (e.g. YOLO) - the rest of the pipeline only needs the list of ground positions.

Projection: pinhole camera pointing straight down. A head pixel is projected onto the
horizontal plane at head height using the drone's current pose.
"""

import math
import time
import numpy as np
import cv2

HEAD_PLANE_Z = 1.05          # m, approximate height of the visible top of a head


class DensityPerceptionEngine:
    def __init__(self, frame_width: int = 640, frame_height: int = 480, fovy_deg: float = 78.0):
        self.frame_w = frame_width
        self.frame_h = frame_height
        self.f = (frame_height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)   # focal length, px
        self.head_lower = np.array([18, 120, 150], dtype=np.uint8)               # yellow heads (HSV)
        self.head_upper = np.array([38, 255, 255], dtype=np.uint8)
        self.kernel = np.ones((3, 3), np.uint8)

    # ---------------------------------------------------------------- geometry
    def pixel_to_world(self, u, v, drone_pos):
        h = max(0.5, float(drone_pos[2]) - HEAD_PLANE_Z)
        x = drone_pos[0] + (u - self.frame_w / 2.0) / self.f * h
        y = drone_pos[1] + (self.frame_h / 2.0 - v) / self.f * h
        return np.array([x, y])

    def world_to_pixel(self, xy, drone_pos):
        h = max(0.5, float(drone_pos[2]) - HEAD_PLANE_Z)
        u = self.frame_w / 2.0 + (xy[0] - drone_pos[0]) / h * self.f
        v = self.frame_h / 2.0 - (xy[1] - drone_pos[1]) / h * self.f
        return int(round(u)), int(round(v))

    def footprint(self, drone_pos, margin_px: int = 12):
        """Axis-aligned head-plane rectangle (xmin, xmax, ymin, ymax) the camera reliably sees."""
        x0, y1 = self.pixel_to_world(margin_px, margin_px, drone_pos)
        x1, y0 = self.pixel_to_world(self.frame_w - margin_px, self.frame_h - margin_px, drone_pos)
        return (x0, x1, y0, y1)

    # ---------------------------------------------------------------- per-frame analysis
    def analyze_frame(self, rgb_frame, drone_pos, simulate_dropout: bool = False):
        """
        Returns dict:
            detections   (N, 2) world xy of detected heads
            pixels       list of (u, v) image positions, aligned with detections
            footprint    observed rectangle on the head plane
            vision_confidence  0..1 image-health score
            bgr          frame as BGR for annotation
            latency_ms
        """
        t0 = time.perf_counter()
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if simulate_dropout:      # optional camera-fault injection to exercise the failsafe
            noise = np.random.randint(0, 256, bgr.shape, dtype=np.uint8)
            bgr = cv2.addWeighted(bgr, 0.10, noise, 0.90, 0)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        confidence = 0.12 if cv2.Laplacian(gray, cv2.CV_64F).var() > 3200.0 else 0.98

        detections, pixels = [], []
        if confidence >= 0.5:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.head_lower, self.head_upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)

            h = max(0.5, float(drone_pos[2]) - HEAD_PLANE_Z)
            head_area = math.pi * (0.13 / h * self.f) ** 2      # expected pixel area of one head
            for k in range(1, n):
                area = stats[k, cv2.CC_STAT_AREA]
                if area < 0.25 * head_area:
                    continue
                u, v = centroids[k]
                count = max(1, int(round(area / head_area)))    # rare merged heads
                for j in range(count):
                    jitter = (j - (count - 1) / 2.0) * 0.6 * math.sqrt(head_area)
                    pixels.append((u + jitter, v))
                    detections.append(self.pixel_to_world(u + jitter, v, drone_pos))

        return {
            "detections": np.array(detections).reshape(-1, 2),
            "pixels": pixels,
            "footprint": self.footprint(drone_pos),
            "vision_confidence": confidence,
            "bgr": bgr,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }
