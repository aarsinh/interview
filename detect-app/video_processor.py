import cv2
import time
from core.detect import CheatingDetector, detect_gazes
from core.gaze_calibrator import GazeCalibrator
from core.fallback import SimpleGazeDetector

simple_detector = SimpleGazeDetector()

class VideoProcessor:
    def __init__(self):
        self.cheating_detector = CheatingDetector()
        self.gaze_calibrator = GazeCalibrator()
        self.results = []
        self.last_logged_timestamp = -1
        self.frame = 0

    def process_video_file(self, input_path: str):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
        
            try:
                gazes = detect_gazes(frame)
                timestamp = self.frame / fps

                for gaze in gazes:
                    facial_structure = self.gaze_calibrator.analyze_facial_structure(gaze["face"])
                    if facial_structure:
                        self.gaze_calibrator.update_baseline(gaze, facial_structure)
                        calibrated_gaze = self.gaze_calibrator.calibrated_gaze_prediction(gaze, facial_structure)

                        if calibrated_gaze and self.gaze_calibrator.is_calibrated():
                            detection = self.cheating_detector.analyze_gaze_behavior(
                                calibrated_gaze, facial_structure, timestamp
                            )
                            
                            if int(timestamp) != self.last_logged_timestamp:
                                self.results.append({
                                    "timestamp": timestamp,
                                    "alert": detection
                                })

                                self.last_logged_timestamp = int(timestamp)
                                
            except Exception as e:
                print(f"Error at frame {self.frame}: {e}")

            self.frame += 1

        cap.release()
        return self.results
