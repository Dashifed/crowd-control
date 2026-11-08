# 5-Minute Live Competition Pitch Script: Autonomous Crowd-Density Dispersion Fleet

**Track:** Agentverse Robotics Track ("Give an Agent a Body")  
**Total Time:** 5:00 Minutes  

---

## [0:00 – 0:50] 1. The Hook: Crowd Crushes Are a Physics Problem We Can Interrupt Early
> "Every year, festival bottlenecks, stadium exits, and transit platforms experience sudden, unscripted crowd surges. Once local crowd density crosses **6 people per 3-by-3-meter square**, human beings lose the ability to move independently—turning a crowd into a fluid crush hazard within seconds.
>
> Today's venue security relies on fixed wall-mounted cameras and human control rooms. By the time a human spots a crush through an occluded camera angle and radios ground staff, the bottleneck is already locked.
>
> We built an **Autonomous Aerial Crowd-Dispersion Fleet**—giving an intelligent agent not just eyes in the sky, but a physical body that responds in milliseconds, descends directly over a developing crush, projects a high-visibility **Red Dispersion Beacon** onto the ground, and coordinates with its wingman so no sector of the venue is ever left unmonitored."

---

## [0:50 – 1:45] 2. 100% Edge Autonomy: Zero Cloud, Zero Cellular Dependency
> "When 20,000 people pack into a venue, cellular and Wi-Fi networks fail first. An autonomous safety drone cannot wait 800 milliseconds for a cloud VLM API call that might time out while hovering 3 meters above people's heads.
>
> Everything you see in our system runs **100% locally on-device** within an **NVIDIA Jetson Orin Nano** compute budget:
> 1. **Under 4-millisecond Perception (`DensityPerceptionEngine`):** Each drone runs an onboard OpenCV spatial density pipeline. It segments pedestrians from its downward camera, compensates for drone altitude using ground-plane homography, and evaluates a sliding $3\text{m} \times 3\text{m}$ spatial window every frame.
> 2. **Sub-millisecond Decentralized Brain (`FleetCoordinator`):** Our state machine requires under 200 megabytes of RAM and executes in **0.09 milliseconds**—giving us an effective onboard control rate over **150 Hz**, well inside our 30 FPS real-time deadline."

---

## [1:45 – 3:55] 3. Live Demo Walkthrough (`submission_video.mp4`)
*(Play `submission_video.mp4` side-by-side: Left pane = `world_cam` 3rd-person arena view; Right pane = `cam_alpha` live onboard perception HUD)*

* **`00:00 – 00:12` | Nominal Split-Sector Patrol:**
  > "Look at the split-screen video. On the left is our third-person arena view showing two drones: **Drone Alpha** (gold, patrolling West Sector 1) and **Drone Bravo** (blue, patrolling East Sector 2) at 4.2 meters altitude. On the right is Drone Alpha's live onboard camera feed. Pedestrians are tracked with green bounding boxes, and the HUD confirms `CROWD DENSITY: NOMINAL`."

* **`00:12 – 00:22` | Unscripted Crowd Surge & Active Red Beacon Dispersion:**
  > "At **t = 12 seconds**, an unscripted crowd surge pulls pedestrians tightly into the Sector 1 bottleneck. Watch the right screen: Drone Alpha's spatial grid immediately flags a red heat-circle (`3x3m CLUSTER: 10 PEDS`), and the HUD flashes `CRITICAL CRUSH RISK DETECTED - DEPLOYING DISPERSION BEAM`.
  >
  > Look at the physical reaction on the left screen:
  > 1. **Drone Alpha** immediately dives from 4.2 meters down to **3.0 meters** right over the hotspot and fires its downward **Red Warning Beacon**.
  > 2. Pedestrians inside the red ground projection experience an immediate radial dispersion cue, scattering outward from the crush epicenter.
  > 3. **Coordinated Fleet Adaptation:** Notice **Drone Bravo** (the blue drone). The instant Alpha drops low to intervene, Alpha's camera footprint narrows—creating a blind spot. Without any human command, Bravo shifts its orbit to **center-stage at 4.55 meters altitude**, maintaining overwatch across both sectors!"

* **`00:22 – 00:48` | Autonomous Recovery & Second Surge Repeatability:**
  > "Once Drone Alpha's onboard camera verifies that local density has dropped back below threshold, it turns off the red beacon, climbs back to 4.2 meters, and both drones return to split-sector patrol (`RECOVERY` -> `PATROL`). When a second unscripted wave hits at **t = 32 seconds**, the fleet autonomously repeats the entire intervention and blind-spot adaptation cycle."

* **`00:48 – 00:53` | Safety Failsafe Demonstration:**
  > "At **t = 48 seconds**, we inject severe camera occlusion and sensor noise into Drone Alpha's feed. Our Laplacian confidence monitor immediately drops below 50%, triggering the `FAILSAFE` state: the drone shuts off its beacon and climbs straight to a **7.0-meter ceiling hold altitude**, ensuring a visually degraded drone never drops toward a crowd."

---

## [3:55 – 5:00] 4. Honest Engineering Limitations & Real-World Roadmap
> "To deploy this in a real stadium tomorrow, we are transparent about three physical limitations:
> 1. **Flight Endurance:** Multi-rotor hover plus a high-lumen LED array limits flight time to ~22 minutes per battery pack, requiring an automated tether or hot-swap landing pad for 4-hour concerts.
> 2. **Wind & Downwash:** Outdoors in winds above 12 m/s, descending to 3.0 meters is unsafe; in real hardware, altitude remains capped at 6–8 meters using a narrower-angle collimated laser/LED ring instead of descending.
> 3. **Lighting Variations:** While HSV color segmentation gives us deterministic 3-millisecond latency in simulation, real-world stadium lighting requires an INT8-quantized **YOLOv8-Nano + thermal IR camera** on the Jetson Orin Nano GPU—which still fits comfortably inside our 12-millisecond edge budget.
>
> Thank you—we're ready for your questions."
