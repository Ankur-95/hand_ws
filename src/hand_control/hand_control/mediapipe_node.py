#!/usr/bin/env python3
#This line is a shebang that tells the system to use python3 to execute this script.

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
#Defining the path to the hand landmarker model file. It will be stored in the user's home directory under .mediapipe.

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]
# This is a list of tuples representing the connections between hand landmarks for drawing the hand skeleton. 

def download_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    #If the directory already exists, don't throw an error.
    if not os.path.exists(MODEL_PATH):
        print('Downloading hand landmarker model...')
        url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
        urllib.request.urlretrieve(url, MODEL_PATH)
        print('Model downloaded!')
    # This function checks if the hand landmarker model file exists. If it doesn't, it downloads the model from the specified URL and saves it to the defined MODEL_PATH.

# The class below is a class inheriting from Node(Ros2 Node), basically containing  4 functions, with the significance of detecting hand landmarks and publishing them to a ros2 topic.
class HandTrackingNode(Node):
    #First function is the constructor, which initializes the node
    def __init__(self):
        super().__init__('hand_tracking_node')
        #Initializes the ROS2 node with the name 'hand_tracking_node'.
        #super() is used to call the constructor of the parent class Node.
        self.publisher = self.create_publisher(Float32MultiArray, '/hand_landmarks', 10)
        #This line creates a publisher that will publish messages of type Float32MultiArray to the topic '/hand_landmarks' with a queue size of 10.
        download_model()
        # This line calls the download model function to ensure the hand landmarker model is available before proceeding.
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)


        #Configuring the hand landmarker options, including the model path, running mode, number of hands to detect, and confidence thresholds for detection and tracking. 
        #The result_callback is set to the landmark_callback method, which will be called whenever new hand landmarks are detected.
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
        #This line initializes the video capture from the default camera (index 0). It will be used to read frames for hand tracking.
        self.timer = self.create_timer(1/30, self.process_frame)
        #Creating a timer that calls the process_frame method at a rate of 30 frames per sec.
        self.get_logger().info('Hand Tracking Node started!')


    # This landmark_callback is the function that will be called whenever new hand landmark are detected. It takes the result, output image, and timestamp as input parameters. 
    # If hand landmarks are detected, it extracts the coordinates of the landmarks, creates a Float32MultiArray message, and publishes it to the '/hand_landmarks' topic. 
    # It also updates the latest_landmarks attribute with the detected landmarks. If no landmarks are detected, it sets latest_landmarks to None.
    def landmark_callback(self, result, output_image, timestamp_ms):
        if result.hand_landmarks:
            #If hand landmarks are detected, this block of code will execute.
            coords = [] # creating an empty list.
            for landmark in result.hand_landmarks[0]:
                #for each landmark in the detected hand landmarks coordinates are extracted.
                #Where the 0 index is used to access the first detected hand (since num_hands=1).
                coords.extend([landmark.x, landmark.y, landmark.z])
                # Appending the x,y andz coordinates of the landmark to the coords list.
            msg = Float32MultiArray() # Creating a new empty Float32MultiArray message.
            msg.data = coords # Assigning the list of coordinates to the data field of the message. All 63 values.
            self.publisher.publish(msg) # Publishing the message to the '/hand_landmarks' topic.
            self.latest_landmarks = result.hand_landmarks[0]
            # Updating the latest landmarks attribute with the detected landmarks for using them in process_frame function to draw the hand skeleton.
        else:
            self.latest_landmarks = None
            # If no landmarks are detected, this line sets the latest_landmarks attribute to None.


    # The process_frame method is responsible for capturing frames from the camera, processing them to detect hand landmarks, and displaying the results. 
    # It reads a frame from the video capture, converts it to RGB format, and creates a MediaPipe image. It then calls the detect_async method of the hand landmarker to process the image. 
    # If landmarks are detected, it draws lines and circles on the frame to visualize the hand skeleton. Finally, it displays the processed frame in a window.
    def process_frame(self):
        ret, frame = self.cap.read()
        #This line reads a frame from the video capture. 'ret' is a boolean indicating if the frame was read successfully, and 'frame' is the actual image captured from the camera.
        if not ret:
            return
        self.timestamp += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #This line converts the captured frame from BGR color format(opencv default) to RGB format,  which is required by the Mediapipe handlandmarker.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # This line creates a MediaPipe Image object from the RGB frame. The image format is set to SRGB, and the data is the RGB image.
        self.detector.detect_async(mp_image, self.timestamp)
        # This line calls the detect_async method of the hand landmarker to process the MediaPipe image. 
        # It passes the image and the current timestamp. The result will be handled by the landmark_callback method.
        if self.latest_landmarks: # Checking whether landmark are detected or not.
            h, w = frame.shape[:2] #getting the height and width of the frame to convert normalized landmark coordinate.
            points = [(int(lm.x * w), int(lm.y * h)) for lm in self.latest_landmarks]
            # This line creates a list of tuples containing the pixel coordinates of the detected hand landmarks.
            # The coordinates are normalized (between 0 and 1), so they are multiplied by the width and height of the frame to convert them to pixel values.
            for start, end in CONNECTIONS:
                cv2.line(frame, points[start], points[end], (255, 255, 255), 2)
            for cx, cy in points:
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            # Code responsible for hand skeleton visualization.
            
        cv2.imshow('Hand Tracking', frame) #This line displays the processed frame in a window titled 'Hand Tracking'.
        cv2.waitKey(1) # This line waits for 1 millisecond for a key event. It allows the OpenCV window to refresh and display the updated frame.

    def destroy_node(self):
        self.cap.release() # Releases the webcam.
        cv2.destroyAllWindows() # Closes all OpenCV windows.
        super().destroy_node() # Calls the destroy_node method of the parent class Node to perform any additional cleanup required by the ROS2 node.

def main(args=None):
    rclpy.init(args=args)
    node = HandTrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()