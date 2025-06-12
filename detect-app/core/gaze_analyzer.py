import numpy as np
from datetime import datetime, timedelta
import os
import csv
from collections import deque
import math

DATA_OUTPUT_DIR = "gaze_data"
COLLECT_TRAINING_DATA = True
TIME_SERIES_WINDOW = 30  # seconds for analysis window
SMOOTHING_WINDOW = 7

class TimeSeriesGazeAnalyzer:
    """
    Collects and analyzes gaze data over time for ML training and analysis
    """
    def __init__(self, output_dir=DATA_OUTPUT_DIR):
        self.output_dir = output_dir
        self.gaze_history = deque(maxlen=1000)  # Store last 1000 data points
        self.session_start = datetime.now()
        self.data_file = None
        self.csv_writer = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize CSV file
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.data_file = open(f"{output_dir}/gaze_data_{timestamp}.csv", 'w', newline='')
        
        fieldnames = [
            'timestamp', 'frame_id', 'raw_yaw', 'raw_pitch', 
            'calibrated_yaw', 'calibrated_pitch', 'confidence',
            'face_x', 'face_y', 'face_width', 'face_height',
            'gaze_point_x', 'gaze_point_y', 'quadrant',
            'smoothed_yaw', 'smoothed_pitch', 'blink_detected',
            'head_movement_velocity', 'gaze_stability_score'
        ]
        
        self.csv_writer = csv.DictWriter(self.data_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        
    def add_gaze_data(self, gaze_data, frame_id=0):
        """Add timestamped gaze data point"""
        current_time = datetime.now()
        timestamp = current_time.isoformat()
        
        # Calculate additional metrics
        head_velocity = self._calculate_head_velocity(gaze_data)
        stability_score = self._calculate_stability_score()
        
        data_point = {
            'timestamp': timestamp,
            'frame_id': frame_id,
            'raw_yaw': gaze_data.get('raw_yaw', 0),
            'raw_pitch': gaze_data.get('raw_pitch', 0),
            'calibrated_yaw': gaze_data.get('calibrated_yaw', 0),
            'calibrated_pitch': gaze_data.get('calibrated_pitch', 0),
            'confidence': gaze_data.get('confidence', 0),
            'face_x': gaze_data.get('face_x', 0),
            'face_y': gaze_data.get('face_y', 0),
            'face_width': gaze_data.get('face_width', 0),
            'face_height': gaze_data.get('face_height', 0),
            'gaze_point_x': gaze_data.get('gaze_point_x', 0),
            'gaze_point_y': gaze_data.get('gaze_point_y', 0),
            'quadrant': gaze_data.get('quadrant', 'unknown'),
            'smoothed_yaw': gaze_data.get('smoothed_yaw', 0),
            'smoothed_pitch': gaze_data.get('smoothed_pitch', 0),
            'blink_detected': gaze_data.get('blink_detected', False),
            'head_movement_velocity': head_velocity,
            'gaze_stability_score': stability_score
        }
        
        self.gaze_history.append(data_point)
        self.csv_writer.writerow(data_point)
        self.data_file.flush()  # Ensure data is written immediately
        
    def _calculate_head_velocity(self, current_data):
        """Calculate head movement velocity"""
        if len(self.gaze_history) < 2:
            return 0.0
            
        prev_data = self.gaze_history[-1]
        
        yaw_diff = current_data.get('raw_yaw', 0) - prev_data.get('raw_yaw', 0)
        pitch_diff = current_data.get('raw_pitch', 0) - prev_data.get('raw_pitch', 0)
        
        return math.sqrt(yaw_diff**2 + pitch_diff**2)
    
    def _calculate_stability_score(self, window_size=10):
        """Calculate gaze stability score over recent frames"""
        if len(self.gaze_history) < window_size:
            return 1.0
            
        recent_data = list(self.gaze_history)[-window_size:]
        
        yaw_values = [d.get('calibrated_yaw', 0) for d in recent_data]
        pitch_values = [d.get('calibrated_pitch', 0) for d in recent_data]
        
        yaw_stability = 1.0 / (1.0 + np.std(yaw_values))
        pitch_stability = 1.0 / (1.0 + np.std(pitch_values))
        
        return (yaw_stability + pitch_stability) / 2
    
    def get_time_series_analysis(self, window_seconds=TIME_SERIES_WINDOW):
        """Generate time series analysis for recent data"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(seconds=window_seconds)
        
        recent_data = [
            d for d in self.gaze_history 
            if datetime.fromisoformat(d['timestamp']) > cutoff_time
        ]
        
        if len(recent_data) < 10:
            return None
            
        # Calculate statistics
        calibrated_yaws = [d['calibrated_yaw'] for d in recent_data]
        calibrated_pitches = [d['calibrated_pitch'] for d in recent_data]
        confidences = [d['confidence'] for d in recent_data]
        
        analysis = {
            'window_seconds': window_seconds,
            'data_points': len(recent_data),
            'avg_yaw': np.mean(calibrated_yaws),
            'avg_pitch': np.mean(calibrated_pitches),
            'yaw_std': np.std(calibrated_yaws),
            'pitch_std': np.std(calibrated_pitches),
            'avg_confidence': np.mean(confidences),
            'gaze_range_yaw': np.max(calibrated_yaws) - np.min(calibrated_yaws),
            'gaze_range_pitch': np.max(calibrated_pitches) - np.min(calibrated_pitches),
            'stability_score': np.mean([d['gaze_stability_score'] for d in recent_data])
        }
        
        return analysis
    
    def close(self):
        """Close data file"""
        if self.data_file:
            self.data_file.close()


class GazeSmoothing:
    def __init__(self, window_size=SMOOTHING_WINDOW):
        self.window_size = window_size
        self.yaw_history = deque(maxlen=window_size)
        self.pitch_history = deque(maxlen=window_size)
        self.gaze_point_history = deque(maxlen=window_size)
    
    def add_gaze(self, yaw, pitch, gaze_point):
        self.yaw_history.append(yaw)
        self.pitch_history.append(pitch)
        self.gaze_point_history.append(gaze_point)
    
    def get_smoothed_gaze(self):
        if len(self.yaw_history) == 0:
            return None, None, None
        
        # Calculate weighted average (more recent values have higher weight)
        weights = np.linspace(0.5, 1.0, len(self.yaw_history))
        weights = weights / np.sum(weights)
        
        smoothed_yaw = np.average(list(self.yaw_history), weights=weights)
        smoothed_pitch = np.average(list(self.pitch_history), weights=weights)
        
        # Smooth gaze point
        gaze_points = np.array(list(self.gaze_point_history))
        smoothed_gaze_point = np.average(gaze_points, axis=0, weights=weights).astype(int)
        
        return smoothed_yaw, smoothed_pitch, tuple(smoothed_gaze_point)