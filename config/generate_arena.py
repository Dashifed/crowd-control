#!/usr/bin/env python3
"""
Generates `config/arena.xml`: a 20 m x 15 m indoor concert floor for 100 pedestrians.

Venue layout (world frame, metres):
    x in [-10, 10], y in [-7.5, 7.5]
    Stage on the WEST wall (x < -8.6), crowd barrier at x = -8.4.
    Bar counter on the EAST wall.

Sizing rationale: 100 people on ~270 m2 of usable floor is ~0.37 people/m2 on average
(comfortable), but the strip in front of the stage centre is only a few m2, so fans
pressing forward can push local density past 4-5 people/m2 if nothing intervenes.

Pedestrians are 70 kg cylinders (r = 0.22 m, shoulder width ~0.44 m) driven by
velocity actuators on two slide joints. Heads are a single distinctive colour so the
downward cameras can count heads (bodies merge in dense crowds; heads stay separated).

Run:  python config/generate_arena.py
"""

import os
import numpy as np

NUM_PEDS = 100
VENUE_X = 10.0          # half-length (x in [-10, 10])
VENUE_Y = 7.5           # half-width  (y in [-7.5, 7.5])
STAGE_X = -8.6          # stage front edge
BARRIER_X = -8.4        # crowd barrier line
PED_RADIUS = 0.22
HEAD_RGBA = "1.0 0.85 0.10 1"
SHIRT_COLOURS = [
    "0.15 0.25 0.60 1", "0.35 0.20 0.50 1", "0.20 0.20 0.24 1",
    "0.10 0.40 0.30 1", "0.55 0.55 0.62 1", "0.10 0.45 0.60 1",
]


def look_at_xyaxes(pos, target, up=(0.0, 0.0, 1.0)):
    """MuJoCo camera xyaxes for a camera at `pos` looking at `target`."""
    pos, target, up = map(np.asarray, (pos, target, up))
    d = target - pos
    d = d / np.linalg.norm(d)
    x = np.cross(d, up)
    x = x / np.linalg.norm(x)
    y = np.cross(x, d)
    return " ".join(f"{v:.4f}" for v in np.concatenate([x, y]))


def drone_xml(name, rgba, rotor_rgba, start):
    # Visual-only geometry (contype/conaffinity 0): drawn about 1.8x real size and
    # self-lit (emissive material) so the drones stand out in the overview camera.
    # Nothing here is below the downward camera, so the drone's own view is unchanged.
    # The spotlight is drawn as a red ring on the floor rather than a MuJoCo <light>:
    # on macOS, spot lights attached to the drones made the drones render black.
    return f"""
        <body name="drone_{name}" mocap="true" pos="{start}">
            <geom name="{name}_core" type="box" size="0.42 0.42 0.10" material="{name}_body_mat" contype="0" conaffinity="0"/>
            <geom name="{name}_arm1" type="capsule" fromto="-0.65 -0.65 0 0.65 0.65 0" size="0.05" material="{name}_body_mat" contype="0" conaffinity="0"/>
            <geom name="{name}_arm2" type="capsule" fromto="-0.65 0.65 0 0.65 -0.65 0" size="0.05" material="{name}_body_mat" contype="0" conaffinity="0"/>
            <geom name="{name}_r1" type="cylinder" pos="0.65 0.65 0.06" size="0.30 0.015" material="{name}_rotor_mat" contype="0" conaffinity="0"/>
            <geom name="{name}_r2" type="cylinder" pos="0.65 -0.65 0.06" size="0.30 0.015" material="{name}_rotor_mat" contype="0" conaffinity="0"/>
            <geom name="{name}_r3" type="cylinder" pos="-0.65 0.65 0.06" size="0.30 0.015" material="{name}_rotor_mat" contype="0" conaffinity="0"/>
            <geom name="{name}_r4" type="cylinder" pos="-0.65 -0.65 0.06" size="0.30 0.015" material="{name}_rotor_mat" contype="0" conaffinity="0"/>
            <geom name="{name}_beacon_bulb" type="sphere" pos="0 0 -0.09" size="0.11" rgba="0.2 0.2 0.2 0.5" contype="0" conaffinity="0"/>
            <camera name="cam_{name}" pos="0 0 -0.10" xyaxes="1 0 0 0 1 0" fovy="78"/>
        </body>
        <body name="beacon_disc_{name}" mocap="true" pos="0 0 -2.0">
            <geom name="disc_{name}_outer" type="cylinder" size="2.0 0.008" contype="0" conaffinity="0" rgba="1.0 0.05 0.05 0.30"/>
            <geom name="disc_{name}_inner" type="cylinder" size="1.0 0.010" contype="0" conaffinity="0" rgba="1.0 0.15 0.10 0.40"/>
        </body>"""


def main():
    world_cam = look_at_xyaxes(pos=(3.0, -21.0, 15.0), target=(-1.0, 1.5, 2.2))
    ped_bodies, actuators = [], []
    for i in range(NUM_PEDS):
        shirt = SHIRT_COLOURS[i % len(SHIRT_COLOURS)]
        ped_bodies.append(f"""
        <body name="ped_{i}" pos="0 0 0.45">
            <joint name="ped_{i}_jx" type="slide" axis="1 0 0"/>
            <joint name="ped_{i}_jy" type="slide" axis="0 1 0"/>
            <geom name="ped_{i}_geom" type="cylinder" size="{PED_RADIUS} 0.40" rgba="{shirt}" mass="70.0" friction="0.3 0.005 0.0001"/>
            <geom name="ped_{i}_head" type="sphere" pos="0 0 0.50" size="0.15" rgba="{HEAD_RGBA}" mass="0.1" contype="0" conaffinity="0"/>
        </body>""")
        for ax in ("x", "y"):
            # Velocity servo: force = kv * (v_desired - v), capped at a realistic push force.
            actuators.append(
                f'        <velocity name="act_ped_{i}_{ax}" joint="ped_{i}_j{ax}" kv="140" '
                f'ctrlrange="-2.0 2.0" forcerange="-260 260"/>')

    hx, hy = VENUE_X, VENUE_Y
    stage_cx = (-hx + STAGE_X) / 2.0
    stage_hx = (STAGE_X + hx) / 2.0
    xml = f"""<mujoco model="crowdflow_concert_floor">
    <!-- AUTO-GENERATED by config/generate_arena.py. Edit the generator, not this file. -->
    <option timestep="0.005" gravity="0 0 -9.81"/>

    <visual>
        <headlight ambient="0.38 0.38 0.42" diffuse="0.45 0.45 0.50" specular="0.1 0.1 0.12"/>
        <rgba haze="0.10 0.12 0.16 1"/>
        <global offwidth="640" offheight="480"/>
        <quality shadowsize="0"/>
    </visual>

    <asset>
        <texture name="grid_tex" type="2d" builtin="checker" rgb1="0.15 0.17 0.22" rgb2="0.19 0.22 0.27" width="512" height="512"/>
        <material name="floor_mat" texture="grid_tex" texrepeat="10 8" reflectance="0"/>
        <material name="wall_mat" rgba="0.32 0.35 0.42 1"/>
        <material name="stage_mat" rgba="0.25 0.20 0.38 1"/>
        <material name="barrier_mat" rgba="0.55 0.55 0.58 1"/>
        <material name="bar_mat" rgba="0.40 0.28 0.18 1"/>
        <material name="alpha_body_mat" rgba="1.0 0.80 0.10 1" emission="0.9"/>
        <material name="alpha_rotor_mat" rgba="0.2 1.0 0.4 0.85" emission="0.8"/>
        <material name="bravo_body_mat" rgba="0.15 0.65 1.0 1" emission="0.9"/>
        <material name="bravo_rotor_mat" rgba="0.3 0.9 1.0 0.85" emission="0.8"/>
    </asset>

    <worldbody>
        <light name="arena_sun" pos="0 0 14" dir="0 0 -1" directional="true" castshadow="false" diffuse="0.50 0.50 0.55" specular="0.1 0.1 0.1"/>
        <geom name="floor" type="plane" size="{hx + 1} {hy + 1} 0.1" material="floor_mat"/>

        <camera name="world_cam" pos="3.0 -21.0 15.0" xyaxes="{world_cam}" fovy="54"/>

        <!-- Perimeter walls -->
        <geom name="wall_north" type="box" pos="0 {hy + 0.2} 1.5" size="{hx + 0.4} 0.2 1.5" material="wall_mat"/>
        <geom name="wall_south" type="box" pos="0 {-hy - 0.2} 0.6" size="{hx + 0.4} 0.2 0.6" material="wall_mat"/>
        <geom name="wall_east"  type="box" pos="{hx + 0.2} 0 1.5" size="0.2 {hy} 1.5" material="wall_mat"/>
        <geom name="wall_west"  type="box" pos="{-hx - 0.2} 0 1.5" size="0.2 {hy} 1.5" material="wall_mat"/>

        <!-- Stage (west) and front-of-stage crowd barrier -->
        <geom name="stage" type="box" pos="{stage_cx} 0 0.5" size="{stage_hx} 5.5 0.5" material="stage_mat"/>
        <geom name="barrier" type="box" pos="{BARRIER_X + 0.05} 0 0.55" size="0.05 5.8 0.55" material="barrier_mat"/>

        <!-- Bar counter (east) -->
        <geom name="bar" type="box" pos="{hx - 0.5} 0 0.55" size="0.5 3.0 0.55" material="bar_mat"/>
{"".join(drone_xml(n, c, r, s) for n, c, r, s in (
    ("alpha", "0.95 0.78 0.12 1", "0.1 0.95 0.35 0.8", "-5 0 7"),
    ("bravo", "0.15 0.65 0.95 1", "0.2 0.85 1.0 0.8", "5 0 7")))}
{"".join(ped_bodies)}
    </worldbody>

    <actuator>
{chr(10).join(actuators)}
    </actuator>
</mujoco>
"""
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena.xml")
    with open(out, "w") as f:
        f.write(xml)
    print(f"Wrote {out} ({NUM_PEDS} pedestrians)")


if __name__ == "__main__":
    main()
