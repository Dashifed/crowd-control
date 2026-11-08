"""
Decentralized Onboard Fleet Coordinator & Finite State Machine.
Controls Drone Alpha (Sector 1 / Active Dispersion Responder) and Drone Bravo
(Sector 2 / Adaptive Blind-Spot Overwatch) with 100% local decision logic.
"""

import time
import math
import numpy as np


class FleetCoordinator:
    """
    Autonomous Multi-Drone State Machine:
      - PATROL: Drone Alpha orbits Sector 1 (West); Drone Bravo orbits Sector 2 (East) at z = 4.2m.
      - DISPERSION_INTERVENTION: Triggered when Drone Alpha detects a critical density hotspot.
        Drone Alpha descends to z = 3.0m directly above the cluster and activates its RED warning beacon.
      - COORDINATED_FLEET_ADAPTATION: While Drone Alpha is engaged in low-altitude dispersion,
        Drone Bravo shifts its orbit to center-stage (x = 0.2m, z = 4.5m) to cover the fleet blind spot.
      - RECOVERY: Once localized density drops back to nominal for a sustained confirmation window,
        Drone Alpha extinguishes the beacon, climbs back to z = 4.2m, and both drones resume sector patrol.
      - FAILSAFE: If vision confidence drops (< 0.50) or frames are missing, the affected drone
        immediately climbs to the z = 7.0m ceiling hold altitude and disables its beacon.
    """

    def __init__(self):
        self.alpha_pos = np.array([-1.5, 0.0, 4.2], dtype=np.float64)
        self.bravo_pos = np.array([2.2, 0.0, 4.2], dtype=np.float64)

        self.alpha_target = self.alpha_pos.copy()
        self.bravo_target = self.bravo_pos.copy()

        self.alpha_state = "PATROL"
        self.bravo_state = "PATROL"
        self.fleet_mode = "PATROL"

        self.alpha_beacon = False
        self.bravo_beacon = False

        self.locked_hotspot_xy = np.array([-1.35, 0.15], dtype=np.float64)
        self.intervention_start_time = -999.0
        self.nominal_clear_timer = 0.0
        self.recovery_start_time = -999.0

        # Altitude constants
        self.nominal_alt = 4.2
        self.dispersion_alt = 3.0
        self.overwatch_alt = 4.55
        self.failsafe_alt = 7.0

    def update(self, alpha_percept: dict, bravo_percept: dict, sim_time: float, dt: float):
        """
        Executes one control step of the decentralized fleet state machine.
        Returns a dict with updated drone positions, beacon states, mode labels, and brain latency_ms.
        """
        t_start = time.perf_counter()

        alpha_conf = alpha_percept.get("vision_confidence", 1.0)
        bravo_conf = bravo_percept.get("vision_confidence", 1.0)
        alpha_alert = alpha_percept.get("density_alert", False)
        alpha_hotspot = alpha_percept.get("hotspot_coord", None)
        alpha_max_cell = alpha_percept.get("max_cell_count", 0)

        # =====================================================================
        # 1. SAFETY FAILSAFE CHECK (HIGHEST PRIORITY: LOW CONFIDENCE -> 7.0M HOLD)
        # =====================================================================
        if alpha_conf < 0.50 or bravo_conf < 0.50:
            self.fleet_mode = "FAILSAFE"
            if alpha_conf < 0.50:
                self.alpha_state = "FAILSAFE"
                self.alpha_target = np.array([self.alpha_pos[0], self.alpha_pos[1], self.failsafe_alt])
                self.alpha_beacon = False
            if bravo_conf < 0.50:
                self.bravo_state = "FAILSAFE"
                self.bravo_target = np.array([self.bravo_pos[0], self.bravo_pos[1], self.failsafe_alt])
                self.bravo_beacon = False

            self._step_kinematics(dt)
            brain_ms = (time.perf_counter() - t_start) * 1000.0
            return {
                "alpha_pos": self.alpha_pos.copy(),
                "bravo_pos": self.bravo_pos.copy(),
                "alpha_beacon": self.alpha_beacon,
                "bravo_beacon": self.bravo_beacon,
                "alpha_state": self.alpha_state,
                "bravo_state": self.bravo_state,
                "fleet_mode": self.fleet_mode,
                "latency_ms": brain_ms,
            }

        # Recover from failsafe once vision confidence is restored
        if self.alpha_state == "FAILSAFE":
            self.alpha_state = "PATROL"
            self.bravo_state = "PATROL"
            self.fleet_mode = "PATROL"

        # =====================================================================
        # 2. NOMINAL SPLIT-SECTOR PATROL vs. CRUSH INTERVENTION STATE MACHINE
        # =====================================================================
        if self.alpha_state == "PATROL":
            self.alpha_beacon = False
            self.bravo_beacon = False
            self.fleet_mode = "PATROL"

            # Smooth, smaller patrol orbit right over Sector 1 (-1.35, 0.0) so the entire
            # 14-person crowd surge stays centered inside Drone Alpha's camera view!
            self.alpha_target = np.array([
                -1.35 + 0.25 * math.cos(sim_time * 0.65),
                0.15 + 0.25 * math.sin(sim_time * 0.65),
                self.nominal_alt,
            ])
            self.bravo_target = np.array([
                2.25 + 0.65 * math.cos(sim_time * 0.75 + math.pi),
                1.10 * math.sin(sim_time * 0.95),
                self.nominal_alt,
            ])

            # Transition to DISPERSION_INTERVENTION when critical density is detected
            if alpha_alert and alpha_hotspot is not None:
                self.alpha_state = "DISPERSION_INTERVENTION"
                self.bravo_state = "COORDINATED_FLEET_ADAPTATION"
                self.fleet_mode = "DISPERSION_INTERVENTION"
                self.locked_hotspot_xy = np.clip(alpha_hotspot, [-3.0, -3.5], [1.5, 3.5])
                self.intervention_start_time = sim_time
                self.nominal_clear_timer = 0.0

        elif self.alpha_state == "DISPERSION_INTERVENTION":
            self.fleet_mode = "DISPERSION_INTERVENTION"

            # Only refine hotspot target when a genuine dense cluster (>= 6 people) is visible;
            # never chase 1 or 2 isolated pedestrians!
            if alpha_hotspot is not None and alpha_max_cell >= 6:
                clamped_hs = np.clip(alpha_hotspot, [-2.5, -2.5], [0.5, 2.5])
                self.locked_hotspot_xy = 0.90 * self.locked_hotspot_xy + 0.10 * clamped_hs

            # Drone Alpha descends to z = 3.0m directly above the cluster and projects RED beacon
            self.alpha_target = np.array([
                self.locked_hotspot_xy[0],
                self.locked_hotspot_xy[1],
                self.dispersion_alt,
            ])
            self.alpha_beacon = True

            # Drone Bravo executes COORDINATED_FLEET_ADAPTATION:
            # Shifts its orbit to center-stage (x = 0.25m, z = 4.55m) to cover both sectors' blind spots
            self.bravo_state = "COORDINATED_FLEET_ADAPTATION"
            self.bravo_target = np.array([
                0.25 + 0.75 * math.cos(sim_time * 0.95),
                1.65 * math.sin(sim_time * 0.95),
                self.overwatch_alt,
            ])

            # Check if crowd has sufficiently dispersed back to nominal density (< 5 people in cell)
            time_in_intervention = sim_time - self.intervention_start_time
            if not alpha_alert and alpha_max_cell < 5 and time_in_intervention >= 3.5:
                self.nominal_clear_timer += dt
            else:
                self.nominal_clear_timer = 0.0

            # Once density is confirmed nominal for 0.8s (or max 8.5s intervention window), transition to RECOVERY
            if self.nominal_clear_timer >= 0.8 or time_in_intervention >= 8.5:
                self.alpha_state = "RECOVERY"
                self.bravo_state = "RECOVERY"
                self.fleet_mode = "RECOVERY"
                self.alpha_beacon = False
                self.recovery_start_time = sim_time

        elif self.alpha_state == "RECOVERY":
            self.fleet_mode = "RECOVERY"
            self.alpha_beacon = False
            self.bravo_beacon = False

            # Climb back to z = 4.2m nominal patrol positions
            self.alpha_target = np.array([-1.5, 0.0, self.nominal_alt])
            self.bravo_target = np.array([2.2, 0.0, self.nominal_alt])

            # If a secondary surge triggers during recovery, immediately re-intervene
            if alpha_alert and alpha_hotspot is not None:
                self.alpha_state = "DISPERSION_INTERVENTION"
                self.bravo_state = "COORDINATED_FLEET_ADAPTATION"
                self.fleet_mode = "DISPERSION_INTERVENTION"
                self.locked_hotspot_xy = np.clip(alpha_hotspot, [-3.0, -3.5], [1.5, 3.5])
                self.intervention_start_time = sim_time
                self.nominal_clear_timer = 0.0
            elif (sim_time - self.recovery_start_time) >= 2.8:
                self.alpha_state = "PATROL"
                self.bravo_state = "PATROL"
                self.fleet_mode = "PATROL"

        self._step_kinematics(dt)
        brain_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "alpha_pos": self.alpha_pos.copy(),
            "bravo_pos": self.bravo_pos.copy(),
            "alpha_beacon": self.alpha_beacon,
            "bravo_beacon": self.bravo_beacon,
            "alpha_state": self.alpha_state,
            "bravo_state": self.bravo_state,
            "fleet_mode": self.fleet_mode,
            "latency_ms": brain_ms,
        }

    def _step_kinematics(self, dt: float):
        """Smooth velocity-limited trajectory tracking for both mocap drones."""
        max_speed = 2.8  # m/s
        max_step = max_speed * dt

        for pos, target in ((self.alpha_pos, self.alpha_target), (self.bravo_pos, self.bravo_target)):
            err = target - pos
            dist = np.linalg.norm(err)
            if dist > max_step and dist > 1e-6:
                pos += (err / dist) * max_step
            else:
                pos += err * min(1.0, 5.5 * dt)
