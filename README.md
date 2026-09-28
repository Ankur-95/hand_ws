Hand WS

ROS 2 + MediaPipe project for detecting hand landmarks from a webcam and using the landmark data for further mapping/control.

Pipeline
Webcam
   ↓
OpenCV
   ↓
MediaPipe Hand Landmarker
   ↓
21 Hand Landmarks
   ↓
/hand_landmarks
   ↓
Mapping Node
Files
mediapipe_node.py — Detects hand landmarks using MediaPipe and publishes their x, y, z coordinates.
mapping_node.py — Receives the landmark data and maps it for the next stage of the project.

---

Main ROS 2 Topic
/hand_landmarks

Message type:

std_msgs/msg/Float32MultiArray

---
Run

Build the workspace:

cd ~/hand_ws
colcon build
source install/setup.bash
cd ~/hand_ws/src


Run MediaPipe node:

ros2 run hand_control mediapipe_node

Run the mapping node in 2nd terminal:

source ~/hand_ws/install/setup.bash
ros2 run two_handlandmarker mapping_node

Run the Rviz on 3rd terminal:
source ~/hand_ws/install/setup.bash
ros2 launch dexhand_description display.launch




