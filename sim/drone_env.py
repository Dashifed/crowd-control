"""
Simulation Environment for Autonomous Crowd-Density Dispersion & Coordinated Fleet.
Loads `config/arena.xml` and manages pedestrian social-force dynamics, unscripted crowd surges,
and active drone red-beacon dispersion physics.
"""

import os
import numpy as np
import mujoco


class CrowdFleetEnv:
    """MuJoCo Simulation Environment for the 2-Drone Crowd Safety Fleet."""

    def __init__(self, xml_path: str = None, frame_width: int = 640, frame_height: int = 480):
        if xml_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            xml_path = os.path.join(base_dir, "config", "arena.xml")

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MJCF Arena XML not found at: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.renderer = mujoco.Renderer(self.model, height=frame_height, width=frame_width)

        self.num_pedestrians = 18
        self.sim_dt = self.model.opt.timestep
        self.rng = np.random.default_rng(2026)

        # Resolve Mocap IDs for both drones and ground beacon rings
        self.alpha_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone_alpha")
        self.bravo_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone_bravo")
        self.alpha_mocap_id = self.model.body_mocapid[self.alpha_body_id]
        self.bravo_mocap_id = self.model.body_mocapid[self.bravo_body_id]

        disc_a_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "beacon_disc_alpha")
        disc_b_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "beacon_disc_bravo")
        self.disc_alpha_mocap_id = self.model.body_mocapid[disc_a_body]
        self.disc_bravo_mocap_id = self.model.body_mocapid[disc_b_body]

        # Resolve Spotlight & Visual Cone IDs
        self.light_alpha_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, "light_alpha")
        self.light_bravo_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, "light_bravo")

        self.alpha_bulb_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "alpha_beacon_bulb")
        self.alpha_cone_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "alpha_beam_cone")
        self.bravo_bulb_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "bravo_beacon_bulb")
        self.bravo_cone_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "bravo_beam_cone")

        # Cache Pedestrian Joint & Actuator Addresses
        self.ped_qpos_addrs = []
        self.ped_qvel_addrs = []
        self.ped_body_ids = []
        for i in range(self.num_pedestrians):
            jx_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"ped_{i}_jx")
            jy_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"ped_{i}_jy")
            self.ped_qpos_addrs.append((self.model.jnt_qposadr[jx_id], self.model.jnt_qposadr[jy_id]))
            self.ped_qvel_addrs.append((self.model.jnt_dofadr[jx_id], self.model.jnt_dofadr[jy_id]))
            self.ped_body_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"ped_{i}"))

        # Beacon & Surge State
        self.beacon_active = {"alpha": False, "bravo": False}
        self.surge_active = False
        self.surge_epicenter = np.array([-1.35, 0.15])  # Sector 1 bottleneck crush epicenter
        self.dispersion_radius = 2.65                   # Ground radius R of repulsive beacon

        # Assign nominal wandering waypoints across Sector 1 (West) and Sector 2 (East)
        self.nominal_goals = np.zeros((self.num_pedestrians, 2))
        self._initialize_crowd_positions()

    def _initialize_crowd_positions(self):
        """
        Positions pedestrians around Sector 1 and Sector 2 at well-spaced intervals (~2.2m apart)
        so initial density is strictly NOMINAL (1-2 peds per cell), while close enough to form
        a massive 14-person crowd crush within 3 seconds when `trigger_surge()` is called.
        """
        ring_angles = np.linspace(0, 2.0 * np.pi, 14, endpoint=False)
        for i in range(self.num_pedestrians):
            jx_adr, jy_adr = self.ped_qpos_addrs[i]
            if i < 14:
                # Ring of 14 pedestrians around the Sector 1 bottleneck (-1.4, 0.0) at radius 2.5m - 3.3m
                r = 2.55 + (i % 2) * 0.70
                ang = ring_angles[i]
                px = -1.40 + r * np.cos(ang)
                py = 0.0 + r * np.sin(ang)
                px = np.clip(px, -3.3, 2.2)
                py = np.clip(py, -3.8, 3.8)
            else:
                # 4 pedestrians in East Sector 2 under Drone Bravo
                sec2_coords = [(2.4, 2.0), (2.4, -2.0), (3.8, 1.2), (3.8, -1.2)]
                px, py = sec2_coords[i - 14]

            self.data.qpos[jx_adr] = px
            self.data.qpos[jy_adr] = py
            self.nominal_goals[i] = np.array([px, py])

        self.set_beacon("alpha", False)
        self.set_beacon("bravo", False)
        mujoco.mj_forward(self.model, self.data)

    def trigger_surge(self, active: bool = True):
        """
        Triggers or clears an unscripted crowd surge toward the Sector 1 stage bottleneck,
        causing pedestrians to pack into a high-density crush cluster.
        """
        self.surge_active = active

    def set_beacon(self, drone_id: str, active: bool):
        """
        Toggles the drone's downward RED WARNING BEACON spotlight and ground projection ring.
        drone_id: 'alpha' or 'bravo'
        """
        key = drone_id.lower()
        self.beacon_active[key] = bool(active)

        if key == "alpha":
            light_id = self.light_alpha_id
            bulb_id = self.alpha_bulb_geom
            cone_id = self.alpha_cone_geom
            disc_mocap = self.disc_alpha_mocap_id
            drone_pos = self.data.mocap_pos[self.alpha_mocap_id]
        elif key == "bravo":
            light_id = self.light_bravo_id
            bulb_id = self.bravo_bulb_geom
            cone_id = self.bravo_cone_geom
            disc_mocap = self.disc_bravo_mocap_id
            drone_pos = self.data.mocap_pos[self.bravo_mocap_id]
        else:
            raise ValueError(f"Unknown drone_id '{drone_id}'. Expected 'alpha' or 'bravo'.")

        if active:
            # Bright RED spotlight [1.0, 0.0, 0.0] and visible translucent beam + ground ring
            self.model.light_diffuse[light_id] = np.array([1.0, 0.0, 0.0])
            self.model.light_specular[light_id] = np.array([0.8, 0.1, 0.1])
            self.model.geom_rgba[bulb_id] = np.array([1.0, 0.0, 0.0, 1.0])
            self.model.geom_rgba[cone_id] = np.array([1.0, 0.05, 0.05, 0.22])
            self.data.mocap_pos[disc_mocap] = np.array([drone_pos[0], drone_pos[1], 0.008])
        else:
            # Dim/Off spotlight and hide ground projection disc beneath floor
            self.model.light_diffuse[light_id] = np.array([0.05, 0.05, 0.05])
            self.model.light_specular[light_id] = np.array([0.0, 0.0, 0.0])
            self.model.geom_rgba[bulb_id] = np.array([0.2, 0.2, 0.2, 0.5])
            self.model.geom_rgba[cone_id] = np.array([1.0, 0.0, 0.0, 0.0])
            self.data.mocap_pos[disc_mocap] = np.array([drone_pos[0], drone_pos[1], -2.0])

    def get_pedestrian_positions(self) -> np.ndarray:
        """Returns (N, 2) array of current (x, y) world coordinates for all pedestrians."""
        coords = np.zeros((self.num_pedestrians, 2))
        for i, body_id in enumerate(self.ped_body_ids):
            coords[i, 0] = self.data.xpos[body_id, 0]
            coords[i, 1] = self.data.xpos[body_id, 1]
        return coords

    def _compute_crowd_controls(self, alpha_pos: np.ndarray, bravo_pos: np.ndarray):
        """
        Computes velocity-steering actuator commands for each pedestrian:
          1. Nominal patrol wandering around arena center OR aggressive surge toward bottleneck.
          2. Active radial dispersion away from any drone projecting an active RED warning beacon.
          3. Social repulsion between pedestrians (reduced during a crush surge to create realistic clustering).
        """
        positions = self.get_pedestrian_positions()
        sim_t = self.data.time

        # If a drone's dispersion beacon is active over the surge zone, it breaks the surge!
        alpha_dispersing = self.beacon_active["alpha"]
        bravo_dispersing = self.beacon_active["bravo"]

        for i in range(self.num_pedestrians):
            pos = positions[i]
            ctrl_vec = np.zeros(2)

            # --- 1. Active Drone Beacon Dispersion (Smooth Outward Walk, No Panic/Jitter) ---
            under_active_beacon = False
            for is_active, d_pos in ((alpha_dispersing, alpha_pos), (bravo_dispersing, bravo_pos)):
                if is_active:
                    beam_xy = d_pos[:2]
                    away_vec = pos - beam_xy
                    dist_beam = np.linalg.norm(away_vec)
                    if dist_beam < self.dispersion_radius:
                        under_active_beacon = True
                        # Walk calmly outward toward assigned dispersed home position
                        to_home = self.nominal_goals[i] - pos
                        dist_home = np.linalg.norm(to_home)
                        if dist_home > 0.25:
                            unit_out = to_home / dist_home
                        elif dist_beam > 1e-3:
                            unit_out = away_vec / dist_beam
                        else:
                            angle = (2.0 * np.pi * i) / self.num_pedestrians
                            unit_out = np.array([np.cos(angle), np.sin(angle)])
                        # Smooth, steady walking speed (max 1.45 m/s effort) so they never panic or jitter
                        ctrl_vec = unit_out * min(1.45, 0.45 + 0.55 * dist_home)

            # --- 2. Surge vs. Nominal Circulation (when not actively dispersing under beam) ---
            if not under_active_beacon:
                if self.surge_active and i < 14:
                    # 14 pedestrians converge steadily into a tight 14-person crowd at the stage bottleneck
                    to_epi = self.surge_epicenter - pos
                    dist_epi = np.linalg.norm(to_epi) + 1e-6
                    ctrl_vec += (to_epi / dist_epi) * min(1.85, dist_epi * 1.45)
                else:
                    # Gentle, calm strolling around well-spaced nominal stations
                    phase = sim_t * 0.35 + i * 0.85
                    wander_offset = np.array([0.28 * np.cos(phase), 0.28 * np.sin(phase * 0.9)])
                    target = self.nominal_goals[i] + wander_offset
                    to_target = target - pos
                    dist_t = np.linalg.norm(to_target) + 1e-6
                    ctrl_vec += (to_target / dist_t) * min(1.10, dist_t * 0.90)

            # --- 3. Smooth Pedestrian-to-Pedestrian Social Spacing ---
            if not (self.surge_active and not alpha_dispersing):
                for j in range(self.num_pedestrians):
                    if i == j:
                        continue
                    diff = pos - positions[j]
                    d = np.linalg.norm(diff)
                    if 1e-4 < d < 0.72:
                        ctrl_vec += (diff / d) * (0.65 * (0.72 - d))

            # --- 4. Soft Boundary Keep-Out from Bottleneck Walls ---
            if pos[0] < -3.2:
                ctrl_vec[0] += 1.2 * (-3.2 - pos[0])
            elif pos[0] > 5.2:
                ctrl_vec[0] -= 1.2 * (pos[0] - 5.2)
            if abs(pos[1]) > 4.2:
                ctrl_vec[1] -= np.sign(pos[1]) * 1.2 * (abs(pos[1]) - 4.2)

            # Write smooth velocity commands to 2-DOF slide actuators
            self.data.ctrl[2 * i] = np.clip(ctrl_vec[0], -2.0, 2.0)
            self.data.ctrl[2 * i + 1] = np.clip(ctrl_vec[1], -2.0, 2.0)

    def step(self, alpha_pos: np.ndarray, bravo_pos: np.ndarray, substeps: int = 1):
        """
        Updates drone mocap positions, sincronizes ground beacon projections,
        and steps the MuJoCo physics simulation by `substeps`.
        """
        alpha_pos = np.asarray(alpha_pos, dtype=np.float64)
        bravo_pos = np.asarray(bravo_pos, dtype=np.float64)

        self.data.mocap_pos[self.alpha_mocap_id] = alpha_pos
        self.data.mocap_pos[self.bravo_mocap_id] = bravo_pos

        # Keep ground projection discs aligned directly beneath active drones
        if self.beacon_active["alpha"]:
            self.data.mocap_pos[self.disc_alpha_mocap_id] = np.array([alpha_pos[0], alpha_pos[1], 0.008])
        if self.beacon_active["bravo"]:
            self.data.mocap_pos[self.disc_bravo_mocap_id] = np.array([bravo_pos[0], bravo_pos[1], 0.008])

        for _ in range(substeps):
            self._compute_crowd_controls(alpha_pos, bravo_pos)
            mujoco.mj_step(self.model, self.data)

    def render_camera(self, cam_name: str) -> np.ndarray:
        """Renders and returns an RGB uint8 numpy array (H, W, 3) from the specified camera."""
        self.renderer.update_scene(self.data, camera=cam_name)
        return self.renderer.render().copy()

    def close(self):
        """Releases offscreen OpenGL rendering resources."""
        self.renderer.close()
