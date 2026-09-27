#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
import os
import urllib.request

MODEL_PATH = os.path.expanduser('~/.mediapipe/hand_landmarker.task')

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def download_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print('Downloading hand landmarker model...')
        url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
        urllib.request.urlretrieve(url, MODEL_PATH)
        print('Model downloaded!')

class HandTrackingNode(Node):
    def __init__(self):
        super().__init__('hand_tracking_node')
        self.publisher = self.create_publisher(Float32MultiArray, '/hand_landmarks', 10)
        download_model()
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            result_callback=self.landmark_callback
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.latest_landmarks = None
        self.timestamp = 0
        self.cap = cv2.VideoCapture(0)
        self.timer = self.create_timer(1/30, self.process_frame)
        self.get_logger().info('Hand Tracking Node started!')

    def landmark_callback(self, result, output_image, timestamp_ms):
        if result.hand_landmarks:
            coords = []
            for landmark in result.hand_landmarks[0]:
                coords.extend([landmark.x, landmark.y, landmark.z])
            msg = Float32MultiArray()
            msg.data = coords
            self.publisher.publish(msg)
            self.latest_landmarks = result.hand_landmarks[0]
        else:
            self.latest_landmarks = None

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        self.timestamp += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.detector.detect_async(mp_image, self.timestamp)
        if self.latest_landmarks:
            h, w = frame.shape[:2]
            points = [(int(lm.x * w), int(lm.y * h)) for lm in self.latest_landmarks]
            for start, end in CONNECTIONS:
                cv2.line(frame, points[start], points[end], (255, 255, 255), 2)
            for cx, cy in points:
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.imshow('Hand Tracking', frame)
        cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = HandTrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()