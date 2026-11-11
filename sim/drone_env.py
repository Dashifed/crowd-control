"""
MuJoCo simulation environment for the CrowdFlow stage-front scenario.

What this module owns:
  * The physical world (config/arena.xml): 100 pedestrians, 2 mocap drones, stage, bar.
  * The crowd behaviour model: social-force-style steering written as velocity commands.
  * How people RESPOND to the drones' outputs (spotlight, announcement). The drones never
    move people directly; they only change what people know, and people decide.
  * Offline evaluation helpers (prefixed `gt_`). These read simulator ground truth and must
    never be passed to the drone brain; they exist only to score a run afterwards.

Scenario: the show starts and fans drift towards the centre of the stage barrier at their
own (random) times. Nobody is scripted to crush; congestion emerges because many people
want the same few square metres. Without intervention, the stage-front centre compresses.
"""

import os
import numpy as np
import mujoco

# Venue geometry (must match config/generate_arena.py)
VENUE_X = 10.0
VENUE_Y = 7.5
BARRIER_X = -8.4
PED_RADIUS = 0.22

# Pedestrian roles
FAN, CASUAL, MOBILE = 0, 1, 2

# Response to drone outputs
SPOTLIGHT_RADIUS = 2.0      # m, ground footprint of the spotlight
AUDIBLE_FULL = 10.0         # m, announcement fully audible within this range
AUDIBLE_ZERO = 30.0         # m, inaudible beyond this range


class CrowdFleetEnv:
    """MuJoCo environment: 100-person concert floor patrolled by drones Alpha and Bravo."""

    DRONES = ("alpha", "bravo")

    def __init__(self, xml_path: str = None, frame_width: int = 640, frame_height: int = 480,
                 seed: int = 2026, render: bool = True):
        if xml_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            xml_path = os.path.join(base_dir, "config", "arena.xml")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MJCF arena not found at {xml_path}. "
                                    "Run `python config/generate_arena.py` first.")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.renderer = mujoco.Renderer(self.model, height=frame_height, width=frame_width) if render else None
        self.sim_dt = self.model.opt.timestep
        self.rng = np.random.default_rng(seed)

        # --- Drones, spotlights, ground rings ---
        self.drone_mocap, self.disc_mocap, self.bulb_geom = {}, {}, {}
        for d in self.DRONES:
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"drone_{d}")
            disc = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"beacon_disc_{d}")
            self.drone_mocap[d] = self.model.body_mocapid[body]
            self.disc_mocap[d] = self.model.body_mocapid[disc]
            self.bulb_geom[d] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{d}_beacon_bulb")
        self.beacon_active = {d: False for d in self.DRONES}

        # --- Pedestrians ---
        n = 0
        while mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"ped_{n}") >= 0:
            n += 1
        self.num_pedestrians = n
        self.qpos_idx = np.zeros((n, 2), dtype=int)
        self.qvel_idx = np.zeros((n, 2), dtype=int)
        for i in range(n):
            for k, ax in enumerate("xy"):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"ped_{i}_j{ax}")
                self.qpos_idx[i, k] = self.model.jnt_qposadr[jid]
                self.qvel_idx[i, k] = self.model.jnt_dofadr[jid]

        self._init_crowd()
        self._control_counter = 0
        self.announcements = []     # log of (time, relief_points)

    # ------------------------------------------------------------------ crowd setup
    def _init_crowd(self):
        n, rng = self.num_pedestrians, self.rng
        # Roles: 60 fans (want the stage-front centre), 28 casual, 12 moving to/from the bar.
        self.role = np.array([FAN] * 60 + [CASUAL] * 28 + [MOBILE] * (n - 88))[:n]
        rng.shuffle(self.role)

        # Non-overlapping random start positions across the floor (people arriving/milling).
        pos = []
        while len(pos) < n:
            p = np.array([rng.uniform(-6.5, 8.3), rng.uniform(-7.0, 7.0)])
            if all(np.linalg.norm(p - q) > 0.95 for q in pos):
                pos.append(p)
        pos = np.array(pos)
        for i in range(n):
            self.data.qpos[self.qpos_idx[i, 0]] = pos[i, 0]
            self.data.qpos[self.qpos_idx[i, 1]] = pos[i, 1]

        self.home = pos.copy()                    # casual "spot" on the floor
        self.target = pos.copy()
        self.speed = rng.uniform(0.9, 1.3, n)     # preferred walking speed (m/s)

        # Fans: when they decide to go forward, and where along the barrier they aim.
        self.fan_start = rng.uniform(3.0, 35.0, n)
        self.fan_aim = np.stack([np.full(n, BARRIER_X + 0.35),
                                 np.clip(rng.normal(0.0, 1.1, n), -5.0, 5.0)], axis=1)
        # Mobile: alternate between the bar and a random floor spot.
        self.next_switch = rng.uniform(5.0, 25.0, n)
        self.at_bar_leg = rng.random(n) < 0.5

        # Behavioural response parameters (stylised; NOT a validated human-compliance model)
        self.responsiveness = rng.uniform(0.35, 0.90, n)
        self.reaction_delay = rng.uniform(1.0, 8.0, n)
        self.holding = np.zeros(n, dtype=bool)        # stopped pushing forward (noticed spotlight)
        self.hold_at = np.full(n, np.inf)             # time the hold takes effect
        self.spot_seen = np.zeros(n, dtype=bool)      # already drew a response to the spotlight
        self.relocate_at = np.full(n, np.inf)         # time a relocation takes effect
        self.relocate_to = np.zeros((n, 2))
        self.relocated = np.zeros(n, dtype=bool)

        for d in self.DRONES:
            self.set_beacon(d, False)
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ drone outputs
    def set_beacon(self, drone_id: str, active: bool):
        """Turns a drone's red spotlight and its ground ring on/off (informational only)."""
        d = drone_id.lower()
        self.beacon_active[d] = bool(active)
        bulb = self.bulb_geom[d]
        # Shown as a red bulb on the drone and a red ring on the floor (see generate_arena.py)
        self.model.geom_rgba[bulb] = [1.0, 0.0, 0.0, 1.0] if active else [0.2, 0.2, 0.2, 0.5]
        self._place_disc(d)

    def _place_disc(self, d):
        p = self.data.mocap_pos[self.drone_mocap[d]]
        z = 0.008 if self.beacon_active[d] else -2.0
        self.data.mocap_pos[self.disc_mocap[d]] = [p[0], p[1], z]

    def announce(self, drone_id: str, hotspot_xy, relief_points, address_radius: float = 5.0):
        """
        Plays a pre-recorded announcement from a drone: "Please make space at the front -
        move to the side areas of the stage." Each person near the hotspot (or a fan still
        heading for it) independently decides whether to comply, based on responsiveness
        and audibility, after a personal reaction delay.
        """
        t = self.data.time
        pos = self.get_pedestrian_positions()
        drone_xy = self.data.mocap_pos[self.drone_mocap[drone_id]][:2]
        relief = np.asarray(relief_points, dtype=float).reshape(-1, 2)
        self.announcements.append((t, relief.copy()))

        dist_drone = np.linalg.norm(pos - drone_xy, axis=1)
        audibility = np.clip((AUDIBLE_ZERO - dist_drone) / (AUDIBLE_ZERO - AUDIBLE_FULL), 0.0, 1.0)
        near_hotspot = np.linalg.norm(pos - np.asarray(hotspot_xy), axis=1) < address_radius
        fan_inbound = (self.role == FAN) & ~self.relocated
        addressed = (near_hotspot | fan_inbound) & ~self.relocated & np.isinf(self.relocate_at)
        comply = addressed & (self.rng.random(self.num_pedestrians) < self.responsiveness * audibility)

        for i in np.flatnonzero(comply):
            if len(relief) == 0:               # "please stop pushing forward"
                self.hold_at[i] = min(self.hold_at[i], t + self.reaction_delay[i])
                continue
            nearest = relief[np.argmin(np.linalg.norm(relief - pos[i], axis=1))]
            self.relocate_to[i] = nearest + self.rng.normal(0.0, 0.8, 2)
            self.relocate_at[i] = t + self.reaction_delay[i]
        return int(comply.sum())

    # ------------------------------------------------------------------ state access
    def get_pedestrian_positions(self) -> np.ndarray:
        return self.data.qpos[self.qpos_idx].copy()

    def get_pedestrian_velocities(self) -> np.ndarray:
        return self.data.qvel[self.qvel_idx].copy()

    # ------------------------------------------------------------------ behaviour model
    def _update_intentions(self, pos):
        t, rng = self.data.time, self.rng

        # Spotlight: people inside the lit circle may notice it and stop pushing forward.
        for d in self.DRONES:
            if not self.beacon_active[d]:
                continue
            c = self.data.mocap_pos[self.drone_mocap[d]][:2]
            inside = np.linalg.norm(pos - c, axis=1) < SPOTLIGHT_RADIUS
            new = inside & ~self.spot_seen
            self.spot_seen |= new
            notice = new & (rng.random(self.num_pedestrians) < self.responsiveness)
            self.hold_at[notice] = t + self.reaction_delay[notice]
        hold_now = (self.hold_at <= t) & ~self.holding
        self.holding |= hold_now
        self.target[hold_now] = pos[hold_now]

        # Announcements take effect after each person's reaction delay.
        move_now = (self.relocate_at <= t) & ~self.relocated
        self.relocated |= move_now
        self.target[move_now] = self.relocate_to[move_now]

        free = ~self.holding & ~self.relocated
        fans = free & (self.role == FAN) & (t >= self.fan_start)
        self.target[fans] = self.fan_aim[fans]

        casual = free & (self.role == CASUAL)
        phase = t * 0.3 + np.arange(self.num_pedestrians) * 0.85
        wander = 0.3 * np.stack([np.cos(phase), np.sin(0.9 * phase)], axis=1)
        self.target[casual] = self.home[casual] + wander[casual]
        fans_waiting = free & (self.role == FAN) & (t < self.fan_start)
        self.target[fans_waiting] = self.home[fans_waiting] + wander[fans_waiting]

        mobile = free & (self.role == MOBILE)
        switch = mobile & (t >= self.next_switch)
        for i in np.flatnonzero(switch):
            self.at_bar_leg[i] = not self.at_bar_leg[i]
            self.next_switch[i] = t + rng.uniform(12.0, 30.0)
            if self.at_bar_leg[i]:
                self.target[i] = [8.6, rng.uniform(-2.6, 2.6)]
            else:
                self.target[i] = [rng.uniform(-4.0, 7.0), rng.uniform(-6.5, 6.5)]

    def _compute_crowd_controls(self):
        pos = self.get_pedestrian_positions()
        self._update_intentions(pos)

        # People relocating after an announcement don't squeeze into a crowd: once they are
        # near their destination and it is already busy around them, they stop where they are.
        diff = pos[:, None, :] - pos[None, :, :]
        d = np.linalg.norm(diff, axis=2) + np.eye(self.num_pedestrians)
        neighbours = (d < 1.0).sum(axis=1)
        settle = (self.relocated & (neighbours >= 4)
                  & (np.linalg.norm(self.target - pos, axis=1) < 2.5))
        self.target[settle] = pos[settle]

        to_t = self.target - pos
        dist = np.linalg.norm(to_t, axis=1, keepdims=True) + 1e-9
        # Fans pressing for the barrier keep pushing; everyone else slows down on arrival.
        pressing = ((self.role == FAN) & ~self.holding & ~self.relocated
                    & (self.data.time >= self.fan_start))[:, None]
        approach = np.where(pressing, 1.0, np.clip(dist / 0.6, 0.0, 1.0))
        v_des = (to_t / dist) * self.speed[:, None] * approach

        # Personal-space repulsion (fans pressing forward tolerate much less space).
        overlap = np.clip(0.85 - d, 0.0, None)
        np.fill_diagonal(overlap, 0.0)
        push = ((diff / d[..., None]) * overlap[..., None]).sum(axis=1)
        weight = np.where(pressing[:, 0], 0.25, 1.2)[:, None]
        v_des += weight * push

        # Soft keep-out from walls, bar and barrier.
        v_des[:, 0] += np.clip((BARRIER_X + 0.3) - pos[:, 0], 0, None) * 1.5
        v_des[:, 0] -= np.clip(pos[:, 0] - (VENUE_X - 1.3), 0, None) * 1.5
        v_des[:, 1] -= np.sign(pos[:, 1]) * np.clip(np.abs(pos[:, 1]) - (VENUE_Y - 0.4), 0, None) * 1.5

        self.data.ctrl[: 2 * self.num_pedestrians] = np.clip(v_des, -2.0, 2.0).reshape(-1)

    # ------------------------------------------------------------------ stepping/rendering
    def step(self, drone_positions: dict, substeps: int = 1):
        """Moves drones to their commanded setpoints and advances physics by `substeps`."""
        for d, p in drone_positions.items():
            self.data.mocap_pos[self.drone_mocap[d]] = np.asarray(p, dtype=np.float64)
            self._place_disc(d)
        for _ in range(substeps):
            if self._control_counter % 2 == 0:      # crowd steering at 100 Hz
                self._compute_crowd_controls()
            self._control_counter += 1
            mujoco.mj_step(self.model, self.data)

    def render_camera(self, cam_name: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=cam_name)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    # ------------------------------------------------------------------ OFFLINE EVALUATION ONLY
    def gt_peak_local_density(self, radius: float = 1.0):
        """
        Ground-truth peak density (people/m2) in any circle of `radius` centred on a person,
        plus where it occurs. Evaluation only - never feed this to the drones.
        """
        pos = self.get_pedestrian_positions()
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        counts = (d < radius).sum(axis=1)
        i = int(np.argmax(counts))
        return counts[i] / (np.pi * radius ** 2), pos[i]
