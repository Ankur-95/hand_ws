#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
import numpy as np
import math

class MappingNode(Node):
    def __init__(self):
        super().__init__('mapping_node') # Initialize the ROS2 node with the name 'mapping_node'.
        self.subscription = self.create_subscription(
            Float32MultiArray, '/hand_landmarks', self.landmark_callback, 10)
        # This line here creates a subscription to the topic '/hand_landmarks'  with message type and queue size(10). 
        # The callback function landmark_callback will be called whenever a new message is received on this topic.
        self.publisher = self.create_publisher(JointState, '/joint_states', 10)
        # Creates a publisher that will publish messages of type JointState to the topic '/joint_states' with a queue size of 10.
        
        # Creates a list containing the names of your robot-hand joints.
        self.joint_names = [
            'thumb_pitch', 'thumb_knuckle', 'thumb_tip', # 3 joints for the thumb
            'index_pitch', 'index_knuckle', 'index_tip', # 3 joints for the index finger
            'middle_pitch', 'middle_knuckle', 'middle_tip', # ==
            'ring_pitch', 'ring_knuckle', 'ring_tip',# ==
            'pinky_pitch', 'pinky_knuckle', 'pinky_tip',# ==
            'thumb_yaw', 'index_yaw', 'middle_yaw', 'ring_yaw', 'pinky_yaw',# 5 joints for yaw movements of each finger
            'wrist_pitch_lower', 'wrist_pitch_upper', 'wrist_yaw', 'thumb_roll' # 4 joints for wrist and thumb roll movements
        ]
        self.alpha = 0.3 # This is exponential smoothing factor for joint positions. Decides how much weight to give to new positions vs previous positions. here 0.3 means 30% weight to new positions and 70% to previous positions.
        # Increasing alpha makes the system more responsive to changes in hand position, while decreasing it makes the system smoother and less sensitive to noise. 
        # To lesser the jitter, you can decrease alpha, but it will also make the system less responsive to rapid changes in hand position.
        self.smoothed_positions = None # will hold the smoothed joint positions, initialized to None, meaning there will be no previously held positions.
        self.get_logger().info('Mapping Node started!')# log msg.


    # The following function retrieves landmark from available landmarks based on the index provided. As a 3d point in space.
    def get_landmark(self, landmarks, index): # Here self is that particular instance of the class, landmarks is the list of landmarks received from the hand tracking node and index is the index of the landmark we want to retrieve. 
        i = index * 3 # Since each landmark occupies three positions.
        return np.array([landmarks[i], landmarks[i+1], landmarks[i+2]]) # Returns a array containing the x, y, z coordinates of the specified landmark.

    # Function for calculating angle between three points. 
    def angle_between(self, a, b, c): # Here a, b, c are the three points in 3D space represented as numpy arrays. The function calculates the angle at point b formed by the line segments ab and bc.
        ba = a - b 
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6) # Calculating the cosine of the angle using dot product formula. 
        cosine = np.clip(cosine, -1.0, 1.0) #Clipping the cosine value to be in the range (-1, 1) to avoid numerical errors that might cause the arccos function to return NaN.
        return math.acos(cosine) # Returning the angle in radians by taking the arccosine of the cosine value, for the ease of use in further calculations, as angles in radians are often more convenient in mathematical computations.

    # The following function calculates the angles for a finger based on the landmarks of its joints. 
    def finger_angles(self, landmarks, p1, p2, p3, p4): # Here self is the instance of the class, landmarks is the list of landmarks received from the hand tracking node, and p1, p2, p3, p4 are the indices of the landmarks corresponding to the joints of a finger. 
        a = self.get_landmark(landmarks, p1) # Retrieves the coordinates of the first joint of the finger using the get_landmark function.
        b = self.get_landmark(landmarks, p2) # Retrieves the coordinates of the second joint of the finger.
        c = self.get_landmark(landmarks, p3)
        d = self.get_landmark(landmarks, p4)
        pitch   = max(0.0, math.pi - self.angle_between(a, b, c)) # Calculating pitch angle at b joint. And it is being substracted from pi to get actual bend angle rather than the angle between the segments.
        knuckle = max(0.0, math.pi - self.angle_between(b, c, d)) # Same for knuckle angle.
        tip     = max(0.0, math.pi - (knuckle * 0.7)) # Calculating tip angle based on the knuckle angle. Instead of calculating tip angle directly we are estimating it as a fraction of knuckle angle.
        return pitch, knuckle, tip

    def landmark_callback(self, msg):
        landmarks = msg.data
        if len(landmarks) != 63:
            return
        thumb_p,  thumb_k,  thumb_t  = self.finger_angles(landmarks, 0, 1, 2, 3) # Allocating the angles for each finger by calling the finger angle function with the appropriate landmark indices for each finger.
        index_p,  index_k,  index_t  = self.finger_angles(landmarks, 5, 6, 7, 8)
        middle_p, middle_k, middle_t = self.finger_angles(landmarks, 9, 10, 11, 12)
        ring_p,   ring_k,   ring_t   = self.finger_angles(landmarks, 13, 14, 15, 16)
        pinky_p,  pinky_k,  pinky_t  = self.finger_angles(landmarks, 17, 18, 19, 20)
        raw_positions = [ # This is the list of joint positions that will be published to the /joint_states topic. It contains the calculated angles for each finger and some additional joints (like wrist and thumb roll) which are set to 0.0 for now.
            thumb_p, thumb_k, thumb_t,
            index_p, index_k, index_t,
            middle_p, middle_k, middle_t,
            ring_p, ring_k, ring_t,
            pinky_p, pinky_k, pinky_t,
            0.0, 0.0, 0.0, 0.0, 0.0, # These are placeholders for the yaw movements of each finger, which are not being calculated in this implementation.
            0.0, 0.0, 0.0, 0.0
        ]
        if self.smoothed_positions is None: # If this is the first time we are receiving joint positions, we initialize the smoothed_positions with the raw_positions. This is necessary because we need a previous position to apply exponential smoothing.
            self.smoothed_positions = raw_positions # For the first message, there is no previous value to smooth hence we will be using raw values as it is.
        else:
            self.smoothed_positions = [ # This is where the exponential smoothing is applied. Alpha is already defined in the constructor.
                self.alpha * r + (1 - self.alpha) * s
                for r, s in zip(raw_positions, self.smoothed_positions)
            ]
        joint_msg = JointState() # Creates an empty ROS JointState message.
        joint_msg.header.stamp = self.get_clock().now().to_msg() # Sets the timestamp of the message to the current time. This is important for synchronizing data in ROS.
        joint_msg.name = self.joint_names # Sets the names of the joints in the message to the list of joint names defined in the constructor.
        joint_msg.position = self.smoothed_positions # Sets the positions of the joints in the message to the smoothed joint positions calculated earlier.
        self.publisher.publish(joint_msg) # Publishes the JointState message to the /joint_states topic, which can be used by other nodes in the ROS2 system to control the robot hand or for visualization purposes.


# Entry point of your ROS Python program.
def main(args=None):
    rclpy.init(args=args)
    node = MappingNode()
    rclpy.spin(node) 
    node.destroy_node() #Cleans up the ROS node when the program exits.
    rclpy.shutdown()# Shuts down the ROS 2 Python client.

if __name__ == '__main__': # This line checks if the script is being run directly (as opposed to being imported as a module in another script). If it is, it calls the main() function to start the ROS2 node. 
    main() 