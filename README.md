# Autonomous Crowd-Density Dispersion & Coordinated Drone Fleet

[![Track](https://img.shields.io/badge/Agentverse_Hackathon-Robotics_Track-blue)](#)
[![Simulator](https://img.shields.io/badge/Physics-MuJoCo_3.x-orange)](#)
[![Perception](https://img.shields.io/badge/Perception-OpenCV_HSV_%2B_Spatial_Grid-green)](#)
[![Edge Target](https://img.shields.io/badge/Edge_Target-NVIDIA_Jetson_Orin_Nano_(8GB)-76B900)](#)

> **"Give an Agent a Body"** — A 100% on-device, decentralized multi-drone autonomy stack built in pure Python with **MuJoCo**, **OpenCV**, and **NumPy** to detect developing crowd crush hazards from overhead, physically disperse dangerous clusters using an aerial **Red Warning Beacon**, and dynamically adapt fleet patrol orbits to eliminate surveillance blind spots.

---

## Overview

High-density crowd surges at music festivals, stadium exits, and transit bottlenecks turn hazardous within seconds when localized density exceeds **6–8 people per 2.5m x 2.5m cell**. Fixed CCTV cameras suffer from severe perspective occlusion and rely on human control rooms or congested cellular networks that fail during mass gatherings.

This repository implements two complete autonomous crowd-safety simulation pipelines:
1. **Modular Coordinated 2-Drone Dispersion Fleet ([`run_simulation.py`](run_simulation.py)):**
   - **Drone Alpha (Gold)** and **Drone Bravo (Blue)** patrol adjacent venue sectors at `z = 4.2m`.
   - When an unscripted crowd surge creates a high-density crush cluster (>= 8 pedestrians in a 2.5m x 2.5m window), **Drone Alpha** autonomously descends to `z = 3.0m` directly over the hotspot and activates a downward **Red Dispersion Beacon** spotlight, guiding pedestrians smoothly outward until safe spacing is restored.
   - Simultaneously, **Drone Bravo** executes **Coordinated Fleet Adaptation**, shifting its orbit to center-stage at `z = 4.55m` to cover the surveillance blind spot while Alpha is engaged at low altitude.
2. **Single-File Chokepoint Rerouting Baseline ([`run_all.py`](run_all.py)):**
   - Self-contained single-script implementation where an autonomous drone monitors dual exits (`Exit A` and `Exit B`), detects an unscripted barrier blockage at `Exit A`, dynamically redirects crowd flow to `Exit B`, conducts autonomous recon sweeps to detect when `Exit A` reopens, and load-balances pedestrian traffic across both exits.

---

## System Architecture

```text
+-----------------------------------------------------------------------------------+
|                      ONBOARD EDGE COMPUTE (PER DRONE NODE)                        |
|                                                                                   |
|  [Downward RGB Cam] ---> [HSV Centroid + Homography] ---> [2.5m x 2.5m Grid]      |
|     (640x480 @ 30Hz)         (~5.4 ms per Cam)             (Crush Risk >= 8 Peds) |
|                                                                      |            |
|  [7.0m Ceiling Hold] <--- (Conf < 0.50) --- [Decentralized Fleet FSM] <---+       |
|                                                |                  |               |
|                                   (Alpha: Intervention)   (Bravo: Blind-Spot)     |
|                                                v                  v               |
|                                       [Descend z=3.0m +]  [Shift Orbit x=0.25m]   |
|                                       [Red Beacon Beam ]  [Center Overwatch   ]   |
+-----------------------------------------------------------------------------------+
```

### Key Autonomy States ([`brain/fleet_planner.py`](brain/fleet_planner.py))
- **`PATROL`:** Split-sector figure-8 surveillance at nominal altitude (`z = 4.2m`).
- **`DISPERSION_INTERVENTION`:** Triggered when `DensityPerceptionEngine` flags a critical cluster (>= 8 pedestrians in a 2.5m x 2.5m cell). Drone Alpha locks onto the cluster centroid, descends to `z = 3.0m`, and activates its downward red spotlight and ground beacon ring.
- **`COORDINATED_FLEET_ADAPTATION`:** Drone Bravo automatically shifts from Sector 2 to a high center-stage orbit (`x = 0.25m, z = 4.55m`) to maintain full-arena visibility while Alpha is at low altitude.
- **`RECOVERY`:** Once onboard perception confirms local density has dropped back to nominal (`< 5` peds/cell), Drone Alpha extinguishes the red beacon and climbs back to `4.2m` before resuming split-sector `PATROL`.
- **`FAILSAFE`:** If camera confidence drops below `50%` (e.g., lens occlusion, smoke, or sensor noise), the drone immediately disables its beacon and climbs to a **`z = 7.0m` ceiling hold altitude** so it never descends blindly above a crowd.

---

## Repository Structure

```text
crowd-management-drone/
├── config/
│   └── arena.xml                 # MuJoCo MJCF model (arena, 18 2-DOF pedestrians, 2 mocap drones & spotlights)
├── sim/
│   └── drone_env.py              # CrowdFleetEnv: social-force crowd physics, surge trigger & beacon repulsion
├── perception/
│   └── density_detector.py       # DensityPerceptionEngine: OpenCV HSV segmentation, homography & HUD overlay
├── brain/
│   └── fleet_planner.py          # FleetCoordinator: 100% local decentralized finite state machine
├── docs/
│   ├── technical_summary.md      # 1-Page Technical Summary deliverable (architecture, Jetson budget, failsafe)
│   └── pitch_script.md           # 5-Minute live competition pitch script & demo walkthrough
├── run_simulation.py             # Main orchestrator for the 2-drone crowd-density dispersion fleet
├── run_all.py                    # Standalone single-file chokepoint blockage & dual-exit load-balancing demo
├── requirements.txt              # Python dependencies (mujoco, opencv-python, numpy)
└── submission_video.mp4          # Generated 60s @ 30 FPS side-by-side competition deliverable video
```

---

## Quickstart & Installation

### 1. Prerequisites
- Python **3.9+** (tested on Windows, Linux, and macOS)
- No GPU or external 3D mesh files required—runs out-of-the-box on CPU.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Coordinated 2-Drone Dispersion Simulation
```bash
python run_simulation.py
```
This runs a 60-second closed-loop simulation (`1,800` frames at `30 FPS`), automatically exports the side-by-side competition video (`world_cam` + `cam_alpha` onboard HUD) to `submission_video.mp4`, and prints the on-device latency benchmark table to the console.

*(Optional)* To run the standalone single-file Chokepoint Rerouting & Load-Balancing simulation:
```bash
python run_all.py
```

---

## On-Device Latency Benchmark

Measured across `N = 1,800` frames (`60s @ 30 FPS`, dual `640x480` camera streams + `200 Hz` MuJoCo physics):

```text
+-----------------------------------+------------+------------+------------+------------+
| On-Device Pipeline Stage          |  Mean (ms) |   Std (ms) |   P95 (ms) |   Max (ms) |
+-----------------------------------+------------+------------+------------+------------+
| Dual-Cam Perception (OpenCV HSV)  |     10.787 |      1.690 |     13.137 |     51.865 |
| Decentralized Brain (Fleet FSM)   |      0.066 |      0.034 |      0.110 |      0.500 |
| Actuation & Physics Step (200Hz)  |     10.265 |      2.058 |     12.845 |     19.763 |
+-----------------------------------+------------+------------+------------+------------+
| Total Onboard Autonomous Loop     |     21.118 |      2.622 |     24.711 |     63.176 |
+-----------------------------------+------------+------------+------------+------------+
Sustained Edge Control Frequency: 47.4 Hz (33.3 ms real-time budget @ 30 FPS)
```

---

## Documentation
- **[1-Page Technical Summary](docs/technical_summary.md):** Problem statement, decentralized architecture diagram, NVIDIA Jetson Orin Nano (8GB) hardware budget, and failsafe analysis.
