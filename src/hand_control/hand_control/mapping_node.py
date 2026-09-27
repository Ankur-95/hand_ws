#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
import numpy as np
import math

class MappingNode(Node):
    def __init__(self):
        super().__init__('mapping_node')
        self.subscription = self.create_subscription(
            Float32MultiArray, '/hand_landmarks', self.landmark_callback, 10)
        self.publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.joint_names = [
            'thumb_pitch', 'thumb_knuckle', 'thumb_tip',
            'index_pitch', 'index_knuckle', 'index_tip',
            'middle_pitch', 'middle_knuckle', 'middle_tip',
            'ring_pitch', 'ring_knuckle', 'ring_tip',
            'pinky_pitch', 'pinky_knuckle', 'pinky_tip',
            'thumb_yaw', 'index_yaw', 'middle_yaw', 'ring_yaw', 'pinky_yaw',
            'wrist_pitch_lower', 'wrist_pitch_upper', 'wrist_yaw', 'thumb_roll'
        ]
        self.alpha = 0.3
        self.smoothed_positions = None
        self.get_logger().info('Mapping Node started!')

    def get_landmark(self, landmarks, index):
        i = index * 3
        return np.array([landmarks[i], landmarks[i+1], landmarks[i+2]])

    def angle_between(self, a, b, c):
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosine = np.clip(cosine, -1.0, 1.0)
        return math.acos(cosine)

    def finger_angles(self, landmarks, p1, p2, p3, p4):
        a = self.get_landmark(landmarks, p1)
        b = self.get_landmark(landmarks, p2)
        c = self.get_landmark(landmarks, p3)
        d = self.get_landmark(landmarks, p4)
        pitch   = max(0.0, math.pi - self.angle_between(a, b, c))
        knuckle = max(0.0, math.pi - self.angle_between(b, c, d))
        tip     = max(0.0, math.pi - (knuckle * 0.7))
        return pitch, knuckle, tip

    def landmark_callback(self, msg):
        landmarks = msg.data
        if len(landmarks) != 63:
            return
        thumb_p,  thumb_k,  thumb_t  = self.finger_angles(landmarks, 0, 1, 2, 3)
        index_p,  index_k,  index_t  = self.finger_angles(landmarks, 5, 6, 7, 8)
        middle_p, middle_k, middle_t = self.finger_angles(landmarks, 9, 10, 11, 12)
        ring_p,   ring_k,   ring_t   = self.finger_angles(landmarks, 13, 14, 15, 16)
        pinky_p,  pinky_k,  pinky_t  = self.finger_angles(landmarks, 17, 18, 19, 20)
        raw_positions = [
            thumb_p, thumb_k, thumb_t,
            index_p, index_k, index_t,
            middle_p, middle_k, middle_t,
            ring_p, ring_k, ring_t,
            pinky_p, pinky_k, pinky_t,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0
        ]
        if self.smoothed_positions is None:
            self.smoothed_positions = raw_positions
        else:
            self.smoothed_positions = [
                self.alpha * r + (1 - self.alpha) * s
                for r, s in zip(raw_positions, self.smoothed_positions)
            ]
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = self.joint_names
        joint_msg.position = self.smoothed_positions
        self.publisher.publish(joint_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MappingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()