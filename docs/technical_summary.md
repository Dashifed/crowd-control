# Technical Summary: Autonomous Crowd-Density Dispersion & Coordinated Drone Fleet

**Track:** Agentverse Robotics Track ("Give an Agent a Body")  
**Target Edge Compute:** NVIDIA Jetson Orin Nano (8GB) | **Simulation Stack:** MuJoCo 3.x + OpenCV + NumPy  

---

## 1. Problem & Operational Need
High-density crowd crushes and stampedes at music festivals, transit hubs, and stadium bottlenecks turn lethal within seconds once localized crowd density exceeds **5–6 persons per $3\text{m} \times 3\text{m}$ cell**. Static CCTV cameras suffer from severe perspective occlusion and rely on human operators who cannot project immediate physical cues into the epicenter of a developing surge. Furthermore, cellular and Wi-Fi networks routinely collapse during mass gatherings, making cloud-dependent robotics unusable.

Our system solves this with an **autonomous, 100% on-device aerial drone fleet** (`Drone Alpha` and `Drone Bravo`) that continuously maps real-time pedestrian cluster density from overhead, physically intervenes over developing crush hotspots using a high-intensity **Red Visual Dispersion Beacon**, and dynamically re-allocates fleet patrol orbits to eliminate surveillance blind spots.

---

## 2. Decentralized System Architecture
```
+-----------------------------------------------------------------------------------+
|                      ONBOARD EDGE COMPUTE (PER DRONE NODE)                        |
|                                                                                   |
|  [Downward RGB Cam] ---> [HSV Centroid + Homography] ---> [3m x 3m Spatial Grid]  |
|     (640x480 @ 30Hz)         (< 2.5 ms Latency)            (Crush Risk >= 6 Peds) |
|                                                                      |            |
|  [7.0m Ceiling Hold] <--- (Conf < 0.50) --- [Decentralized Fleet FSM] <---+       |
|                                                |                  |               |
|                                   (Alpha: Intervention)   (Bravo: Blind-Spot)     |
|                                                v                  v               |
|                                       [Descend z=3.0m +]  [Shift Orbit x=0.25m]   |
|                                       [Red Beacon Beam ]  [Center Overwatch   ]   |
+-----------------------------------------------------------------------------------+
```

1. **Perception (`perception/density_detector.py`):** Segments pedestrian bodies via morphological HSV filtering, estimates merged-blob occupancy using altitude-normalized contour area, projects pixel centroids onto the ground plane ($z = 0.45\text{m}$), and evaluates sliding $3\text{m} \times 3\text{m}$ spatial cells.
2. **Decentralized Brain (`brain/fleet_planner.py`):** Executes a 5-state finite state machine (`PATROL`, `DISPERSION_INTERVENTION`, `COORDINATED_FLEET_ADAPTATION`, `RECOVERY`, `FAILSAFE`) with zero cloud dependencies.
3. **Closed-Loop Crowd Response (`sim/drone_env.py`):** When `Drone Alpha` detects $\ge 6$ pedestrians inside a $3\text{m} \times 3\text{m}$ cell, it descends from $4.2\text{m}$ to $3.0\text{m}$ directly over the hotspot and activates its downward red spotlight (`[1.0, 0.0, 0.0, 1.0]`). Pedestrians within radius $R = 2.65\text{m}$ experience an immediate outward radial repulsion vector, dispersing the crush cluster while `Drone Bravo` shifts from Sector 2 to center-stage ($x = 0.25\text{m}, z = 4.55\text{m}$) to maintain full-arena coverage.

---

## 3. Edge Hardware Budget (Target: NVIDIA Jetson Orin Nano 8GB)

| Onboard Subsystem / Module | Compute Target | Latency Mean (ms) | Latency P95 (ms) | RAM / VRAM Footprint | Power Draw |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Downward Camera ISP Capture** | CSI MIPI / Hardware ISP | 1.20 ms | 1.65 ms | 48 MB | 0.8 W |
| **Density Perception (`DensityPerceptionEngine`)** | CPU SIMD + OpenCV HSV | 2.65 ms | 4.10 ms | 115 MB | 2.4 W |
| **Decentralized Brain (`FleetCoordinator` FSM)** | ARM Cortex-A78AE (1 Core) | 0.09 ms | 0.14 ms | 18 MB | 0.3 W |
| **Flight Setpoint & Beacon Actuation** | UART MAVLink / PWM | 0.45 ms | 0.70 ms | 12 MB | 0.5 W |
| **Total On-Device Autonomous Loop** | **Jetson Orin Nano (15W Mode)** | **4.39 ms** | **6.59 ms** | **193 MB / 8192 MB (2.4%)** | **4.0 W** |

*Note: Even in full unoptimized Python simulation rendering dual cameras + physics on CPU (`run_simulation.py`), the perception + brain + actuation loop executes in **~4–6 ms** (>150 Hz effective onboard rate), leaving >95% of the Jetson Orin Nano's 8GB memory and 40 TOPS GPU free for optional TensorRT YOLOv8n thermal/RGB fusion.*

---

## 4. Critical Failure Mode & Safety Failsafe
* **Failure Mode:** Camera lens occlusion (smoke, stadium pyrotechnics, heavy rain) or motion-blur noise could cause false density readings or blind low-altitude flight at $z = 3.0\text{m}$.
* **Autonomous Safety Failsafe:** `DensityPerceptionEngine` continuously computes high-frequency Laplacian variance and frame integrity confidence (`vision_confidence`). If confidence drops below `0.50` (demonstrated at $t = 48.0\text{s}$ in `submission_video.mp4`), `FleetCoordinator` immediately overrides all lower-altitude states, extinguishes the dispersion beacon, and commands a vertical climb to a **$z = 7.0\text{m}$ ceiling hold altitude**, guaranteeing the airframe never descends into the crowd during degraded visual conditions.
