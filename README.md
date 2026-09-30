# Hand Gesture Controlled Robot Simulation

A ROS 2 project that mirrors real-time hand movements onto a simulated robotic hand (DexHand) using a webcam, MediaPipe, and RViz2.

Move your hand in front of the camera — the robot hand mimics your finger movements live.

---

## Pipeline

```
Webcam → OpenCV → MediaPipe (21 landmarks) → /hand_landmarks topic → Mapping Node → /joint_states topic → RViz2 (DexHand)
```

---

## What I Built

| File | What it does |
|---|---|
| `mediapipe_node.py` | Reads webcam feed, detects 21 hand landmarks using Google's MediaPipe Tasks API, draws skeleton overlay, publishes 63 floats (21 × x,y,z) on `/hand_landmarks` |
| `mapping_node.py` | Subscribes to landmarks, converts finger joint positions into 15 joint angles using vector dot product math, applies exponential smoothing to reduce jitter, publishes `JointState` messages to drive the simulated hand |

---

## ROS 2 Topics

| Topic | Message Type | Direction |
|---|---|---|
| `/hand_landmarks` | `std_msgs/msg/Float32MultiArray` | mediapipe_node → mapping_node |
| `/joint_states` | `sensor_msgs/msg/JointState` | mapping_node → RViz2 |

---

## Tech Stack

- ROS 2 Jazzy + Ubuntu 24.04
- Gazebo Harmonic
- MediaPipe Tasks API (Hand Landmarker)
- OpenCV
- DexHand URDF — [iotdesignshop/dexhand_description](https://github.com/iotdesignshop/dexhand_description)
- MediaPipe Hand Landmarker model — [Google MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)

---

## Setup & Run

### Prerequisites
- ROS 2 Jazzy installed
- Python packages: `pip3 install mediapipe opencv-python 'numpy<2' --break-system-packages`
- ROS bridge: `sudo apt install ros-jazzy-ros-gz ros-jazzy-joint-state-publisher-gui`

### Clone and Build
```bash
mkdir -p ~/hand_ws/src && cd ~/hand_ws/src
git clone https://github.com/Ankur-95/hand_ws.git .
cd ~/hand_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

### Run (3 terminals)

**Terminal 1 — RViz2 with DexHand:**
```bash
source /opt/ros/jazzy/setup.bash && source ~/hand_ws/install/setup.bash
ros2 launch dexhand_description display.launch.py
```

**Terminal 2 — Hand tracking:**
```bash
source /opt/ros/jazzy/setup.bash && source ~/hand_ws/install/setup.bash
ros2 run hand_control mediapipe_node
```

**Terminal 3 — Joint mapping:**
```bash
source /opt/ros/jazzy/setup.bash && source ~/hand_ws/install/setup.bash
ros2 run hand_control mapping_node
```

---

## Known Limitations

- Wrist and yaw joints are currently fixed at 0 — only finger curl is mapped
- Single hand only
- Accuracy depends on lighting conditions(Jitter can be adjusted by varying )
- Z-axis depth from webcam is less reliable than x/y

---



### By
Ankur Rakesh Ujawane   
Robotics & Automation Engineering
