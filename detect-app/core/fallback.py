'''
This file contains the simple detector to fall back upon when the full version isn't available
'''

import cv2
import time
import numpy as np

class SimpleGazeDetector:
    """Simple fallback detector if main modules aren't available"""
    def __init__(self):
        self.frame_count = 0
        self.last_detection_time = time.time()
        
    def detect_gazes(self, frame):
        """Simple face detection using OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        detections = []
        for (x, y, w, h) in faces:
            # Create simple gaze data
            detection = {
                "face": {
                    "x": x + w//2,
                    "y": y + h//2,
                    "width": w,
                    "height": h,
                    "landmarks": [
                        {"x": x + w//4, "y": y + h//3},  # Left eye
                        {"x": x + 3*w//4, "y": y + h//3},  # Right eye
                        {"x": x + w//2, "y": y + 2*h//3},  # Nose
                        {"x": x + w//3, "y": y + 3*h//4},  # Left mouth
                        {"x": x + 2*w//3, "y": y + 3*h//4}  # Right mouth
                    ]
                },
                "yaw": (self.frame_count % 100 - 50) * 0.02,  # Simple animation
                "pitch": 0.1 * np.sin(self.frame_count * 0.1)
            }
            detections.append(detection)
            
        self.frame_count += 1
        return detections