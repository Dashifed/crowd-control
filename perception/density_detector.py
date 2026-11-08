"""
Onboard Edge Perception Engine for Crowd Density Estimation & Crush Hazard Detection.
Processes downward camera frames (`cam_alpha` / `cam_bravo`) in <12ms using OpenCV HSV segmentation,
projects centroids to ground-plane meters, evaluates 3m x 3m spatial grid density, and draws HUD overlays.
"""

import time
import math
import numpy as np
import cv2


class DensityPerceptionEngine:
    """Real-time OpenCV crowd density analyzer and HUD renderer."""

    def __init__(self, frame_width: int = 640, frame_height: int = 480,
                 fovy_deg: float = 78.0, density_threshold: int = 8):
        self.frame_w = frame_width
        self.frame_h = frame_height
        self.fovy_rad = math.radians(fovy_deg)
        self.density_threshold = density_threshold  # Alert only when >= 8 people pack into a 2.5m x 2.5m zone

        # Focal length in pixels from vertical FOV
        self.fy = (self.frame_h / 2.0) / math.tan(self.fovy_rad / 2.0)
        self.fx = self.fy

        # HSV Thresholds for Cyan Pedestrian Bodies (H: 80-105 in OpenCV 0-179 scale)
        self.cyan_lower = np.array([80, 95, 85], dtype=np.uint8)
        self.cyan_upper = np.array([105, 255, 255], dtype=np.uint8)
        self.morph_kernel = np.ones((3, 3), np.uint8)

    def _pixel_to_world_xy(self, u: float, v: float, drone_pos: np.ndarray) -> np.ndarray:
        """
        Projects pixel (u, v) from the downward-facing camera (`cam_alpha` / `cam_bravo`)
        onto the ground plane (z = 0.45m pedestrian torso height).
        Camera frame axes: +X_cam = +X_world (right in image), +Y_cam = +Y_world (up in image),
        -Z_cam = downward (-Z_world).
        """
        height_above_ped = max(1.0, float(drone_pos[2]) - 0.45)
        dx_norm = (u - (self.frame_w / 2.0)) / self.fx
        # Image row v increases downward, which corresponds to -Y in camera/world coordinates
        dy_norm = ((self.frame_h / 2.0) - v) / self.fy

        world_x = float(drone_pos[0]) + dx_norm * height_above_ped
        world_y = float(drone_pos[1]) + dy_norm * height_above_ped
        return np.array([world_x, world_y], dtype=np.float64)

    def _world_xy_to_pixel(self, world_xy: np.ndarray, drone_pos: np.ndarray):
        """Projects a ground-plane (x, y) coordinate back into camera pixel (u, v)."""
        height_above_ped = max(1.0, float(drone_pos[2]) - 0.45)
        dx = float(world_xy[0] - drone_pos[0])
        dy = float(world_xy[1] - drone_pos[1])
        u = int(round((self.frame_w / 2.0) + (dx / height_above_ped) * self.fx))
        v = int(round((self.frame_h / 2.0) - (dy / height_above_ped) * self.fy))
        return u, v

    def analyze_frame(self, rgb_frame: np.ndarray, drone_pos: np.ndarray,
                      drone_label: str = "DRONE ALPHA",
                      fleet_status_text: str = "PATROL",
                      simulate_dropout: bool = False):
        """
        Processes a single downward RGB camera frame.
        Returns a dictionary containing:
            - density_alert (bool): True if localized 3m x 3m density >= threshold
            - hotspot_coord (np.ndarray or None): (x, y) world coordinate of cluster center
            - max_cell_count (int): Peak pedestrian count in a 3m x 3m window
            - total_count (int): Total pedestrians visible in frame
            - vision_confidence (float): [0.0, 1.0] sensor health confidence score
            - annotated_bgr (np.ndarray): Annotated BGR image with bounding boxes, heat circle & HUD
            - latency_ms (float): Frame processing time in milliseconds
        """
        t_start = time.perf_counter()

        if rgb_frame is None or rgb_frame.size == 0:
            blank = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return {
                "density_alert": False,
                "hotspot_coord": None,
                "max_cell_count": 0,
                "total_count": 0,
                "vision_confidence": 0.0,
                "annotated_bgr": blank,
                "latency_ms": latency_ms,
            }

        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Optional simulated sensor glitch / lens occlusion for failsafe testing
        if simulate_dropout:
            noise = np.random.randint(0, 256, bgr.shape, dtype=np.uint8)
            bgr = cv2.addWeighted(bgr, 0.10, noise, 0.90, 0)
            cv2.circle(bgr, (self.frame_w // 2, self.frame_h // 2), 190, (12, 12, 12), -1)

        # Assess image integrity / confidence score
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if simulate_dropout or lap_var > 3200.0:
            vision_confidence = 0.12
        else:
            vision_confidence = 0.98

        annotated = bgr.copy()
        world_points = []
        pixel_boxes = []
        total_pedestrians = 0

        if vision_confidence >= 0.50:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.cyan_lower, self.cyan_upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)

            # Calibrated pixel area of a single 3D pedestrian cylinder+sphere in the downward camera:
            # At z=4.2m (height_above_ped=3.75m), 1 pedestrian is ~2,100 pixels.
            # At z=3.0m (height_above_ped=2.55m), 1 pedestrian is ~4,500 pixels.
            height_above_ped = max(1.5, float(drone_pos[2]) - 0.45)
            single_ped_area = 2250.0 * ((3.75 / height_above_ped) ** 2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 120.0:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)

                # Strictly conservative count: 1 isolated person is ALWAYS 1.
                # Only count multiple people when a contour is genuinely huge (multiple touching cylinders).
                est_count = max(1, min(10, int(round(area / single_ped_area))))
                total_pedestrians += est_count

                M = cv2.moments(cnt)
                if M["m00"] > 1e-5:
                    cu = M["m10"] / M["m00"]
                    cv_y = M["m01"] / M["m00"]
                else:
                    cu = x + w / 2.0
                    cv_y = y + h / 2.0

                w_xy = self._pixel_to_world_xy(cu, cv_y, drone_pos)
                if est_count == 1:
                    world_points.append(w_xy)
                else:
                    # Distribute estimated bodies within the merged cluster footprint
                    for k in range(est_count):
                        ang = (2.0 * math.pi * k) / est_count
                        offset = np.array([0.25 * math.cos(ang), 0.25 * math.sin(ang)])
                        world_points.append(w_xy + offset)

                pixel_boxes.append((x, y, w, h, int(cu), int(cv_y), est_count))

        # --- Spatial 2.5m x 2.5m Sliding Grid Cluster Evaluation ---
        max_cell_count = 0
        hotspot_coord = None

        if len(world_points) > 0:
            pts = np.array(world_points)  # shape (M, 2)
            best_indices = None
            for i in range(len(pts)):
                center = pts[i]
                in_box = (np.abs(pts[:, 0] - center[0]) <= 1.25) & (np.abs(pts[:, 1] - center[1]) <= 1.25)
                count = int(np.sum(in_box))
                if count > max_cell_count:
                    max_cell_count = count
                    best_indices = in_box

            if best_indices is not None and max_cell_count > 0:
                hotspot_coord = np.mean(pts[best_indices], axis=0)

        density_alert = bool(max_cell_count >= self.density_threshold and vision_confidence >= 0.50)

        # --- Draw Visual Overlays: Heat Circle, Bounding Boxes & Centroids ---
        if hotspot_coord is not None and max_cell_count >= 4 and vision_confidence >= 0.50:
            hu, hv = self._world_xy_to_pixel(hotspot_coord, drone_pos)
            # Convert 1.6m ground radius to pixel radius
            height_above_ped = max(1.0, float(drone_pos[2]) - 0.45)
            px_radius = int(round((1.65 / height_above_ped) * self.fx))

            overlay = annotated.copy()
            heat_color = (0, 0, 255) if density_alert else (0, 180, 255)
            cv2.circle(overlay, (hu, hv), px_radius, heat_color, -1)
            cv2.addWeighted(overlay, 0.26 if density_alert else 0.16, annotated,
                            0.74 if density_alert else 0.84, 0, annotated)
            cv2.circle(annotated, (hu, hv), px_radius, heat_color, 2, cv2.LINE_AA)
            cv2.drawMarker(annotated, (hu, hv), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(
                annotated,
                f"3x3m CLUSTER: {max_cell_count} PEDS",
                (max(10, hu - 75), max(95, hv - px_radius - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255) if density_alert else (0, 230, 255),
                2,
                cv2.LINE_AA,
            )

        # Draw individual pedestrian bounding boxes and centroid dots
        for (x, y, w, h, cu, cv_y, est_count) in pixel_boxes:
            box_col = (0, 255, 0) if not density_alert else (0, 200, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_col, 2)
            cv2.circle(annotated, (cu, cv_y), 3, (0, 255, 0), -1)
            lbl = f"P({est_count})" if est_count > 1 else "P"
            cv2.putText(annotated, lbl, (x, max(85, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)

        # --- Draw Top & Bottom Onboard HUD Banners ---
        cv2.rectangle(annotated, (0, 0), (self.frame_w, 74), (16, 18, 22), -1)
        cv2.putText(
            annotated,
            f"{drone_label} ONBOARD CAM | MODE: {fleet_status_text}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (210, 230, 255),
            1,
            cv2.LINE_AA,
        )

        if vision_confidence < 0.50:
            banner_txt = "SENSOR ALERT: LOW VISION CONFIDENCE - CLIMBING TO 7M CEILING"
            banner_col = (0, 165, 255)
        elif density_alert:
            banner_txt = "CRITICAL CRUSH RISK DETECTED - DEPLOYING DISPERSION BEAM"
            banner_col = (40, 40, 255)
        else:
            banner_txt = "CROWD DENSITY: NOMINAL"
            banner_col = (60, 240, 100)

        cv2.putText(annotated, banner_txt, (12, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, banner_col, 2, cv2.LINE_AA)

        # Bottom Telemetry Strip
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        cv2.rectangle(annotated, (0, self.frame_h - 60), (self.frame_w, self.frame_h), (16, 18, 22), -1)
        cv2.putText(
            annotated,
            f"VISIBLE PEDS: {total_pedestrians:02d} | MAX 3x3m CELL: {max_cell_count:02d} (THRESH: {self.density_threshold})",
            (12, self.frame_h - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"ALT: {drone_pos[2]:.1f}m | POS: ({drone_pos[0]:+.1f}, {drone_pos[1]:+.1f}) | "
            f"CONF: {vision_confidence*100:.0f}% | PERCEPT: {latency_ms:.2f}ms",
            (12, self.frame_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (200, 205, 215),
            1,
            cv2.LINE_AA,
        )

        return {
            "density_alert": density_alert,
            "hotspot_coord": hotspot_coord,
            "max_cell_count": max_cell_count,
            "total_count": total_pedestrians,
            "vision_confidence": vision_confidence,
            "annotated_bgr": annotated,
            "latency_ms": latency_ms,
        }
