from config import CheatingDetectionConfig, ALERT_MESSAGES, BEHAVIOR_DESCRIPTIONS
import os
import cv2
import numpy as np
import base64
import requests
from datetime import datetime
import time
from collections import deque
import math

API_KEY = os.environ.get("API_KEY", "")
DISTANCE_TO_OBJECT = 600  # mm
HEIGHT_OF_HUMAN_FACE = 250  # mm
GAZE_DETECTION_URL = (
    "http://127.0.0.1:9001/gaze/gaze_detection?api_key=" + API_KEY
)

# Smoothing parameters
SMOOTHING_WINDOW = 7
MIN_FACE_SIZE = 80
MAX_FACE_SIZE = 250

def detect_gazes(frame: np.ndarray):
    if not API_KEY:
        print('No API Key')
        # Return dummy gaze data for testing when API key is not available
        return [{
            "face": {
                "x": frame.shape[1] // 2,
                "y": frame.shape[0] // 2,
                "width": 150,
                "height": 200,
                "landmarks": [
                    {"x": frame.shape[1] // 2 - 30, "y": frame.shape[0] // 2 - 20},
                    {"x": frame.shape[1] // 2 + 30, "y": frame.shape[0] // 2 - 20},
                    {"x": frame.shape[1] // 2, "y": frame.shape[0] // 2 + 30},
                    {"x": frame.shape[1] // 2 - 20, "y": frame.shape[0] // 2 + 10},
                    {"x": frame.shape[1] // 2 + 20, "y": frame.shape[0] // 2 + 10}
                ]
            },
            "yaw": 0.0,
            "pitch": 0.0
        }]
    
    img_encode = cv2.imencode(".jpg", frame)[1]
    img_base64 = base64.b64encode(img_encode)
    resp = requests.post(
        GAZE_DETECTION_URL,
        json={
            "api_key": API_KEY,
            "image": {"type": "base64", "value": img_base64.decode("utf-8")},
        },
    )
    gazes = resp.json()[0]["predictions"]
    return gazes

class CheatingDetector:
    """
    Real-time cheating detection system for online interviews/exams
    Monitors gaze direction, head pose, and movement patterns
    """
    def __init__(self, config=None):
        if config is None:
            config = CheatingDetectionConfig()
        self.config = config
        
        self.looking_away_start = None
        self.suspicious_events = []
        self.cheating_alerts = []
        self.last_alert_time = 0
        self.head_movement_history = deque(maxlen=50)  # Track recent head movements
        self.gaze_off_screen_history = deque(maxlen=100)
        self.alert_active = False
        self.total_looking_away_time = 0
        self.session_start_time = time.time()
        
        # Pattern detection
        self.repeated_patterns = {
            'downward_looks': 0,
            'side_looks': 0,
            'head_turns': 0
        }
        
        # Scoring system
        self.suspicion_score = 0
        
    def analyze_gaze_behavior(self, calibrated_gaze, facial_structure, frame_timestamp):
        """Analyze gaze and head behavior for cheating indicators"""
        current_time = time.time()
        
        if not calibrated_gaze or not facial_structure:
            return self._create_detection_result(False, "No gaze data", current_time)
            
        yaw = calibrated_gaze['calibrated_yaw']
        pitch = calibrated_gaze['calibrated_pitch']
        confidence = calibrated_gaze['confidence']
        
        # Only analyze if confidence is reasonable
        if confidence < 0.3:
            return self._create_detection_result(False, "Low confidence gaze data", current_time)
        
        # Check if looking away from screen
        is_looking_away = self._is_looking_away(yaw, pitch)
        
        # Track looking away duration
        if is_looking_away:
            if self.looking_away_start is None:
                self.looking_away_start = current_time
            looking_away_duration = current_time - self.looking_away_start
        else:
            if self.looking_away_start is not None:
                # Was looking away, now back to screen
                away_duration = current_time - self.looking_away_start
                self.total_looking_away_time += away_duration
                self.looking_away_start = None
                looking_away_duration = 0
            else:
                looking_away_duration = 0
        
        # Analyze head movement patterns
        head_movement_velocity = self._calculate_head_movement(yaw, pitch)
        
        # Update pattern counters
        self._update_pattern_detection(yaw, pitch, current_time)
        
        # Calculate suspicion score
        suspicion_increase = self._calculate_suspicion_increase(
            is_looking_away, looking_away_duration, head_movement_velocity, yaw, pitch
        )
        self.suspicion_score = min(self.config.MAX_SUSPICION_SCORE, 
                                 max(0, self.suspicion_score + suspicion_increase))
        
        # Determine alert level
        alert_level = self._determine_alert_level(looking_away_duration, current_time)
        
        return self._create_detection_result(
            is_looking_away, alert_level, current_time, {
                'looking_away_duration': looking_away_duration,
                'total_away_time': self.total_looking_away_time,
                'suspicion_score': self.suspicion_score,
                'head_movement_velocity': head_movement_velocity,
                'yaw': yaw,
                'pitch': pitch,
                'confidence': confidence
            }
        )
    
    def _is_looking_away(self, yaw, pitch):
        """Determine if gaze is away from screen"""
        # Looking down (reading notes)
        if pitch > self.config.DOWNWARD_GAZE_ANGLE:
            return True
        
        # Looking to sides (other screens/people)
        if abs(yaw) > self.config.SIDE_GAZE_ANGLE:
            return True
            
        return False
    
    def _calculate_head_movement(self, yaw, pitch):
        """Calculate head movement velocity"""
        current_pose = (yaw, pitch)
        self.head_movement_history.append(current_pose)
        
        if len(self.head_movement_history) < 2:
            return 0.0
        
        prev_pose = self.head_movement_history[-2]
        velocity = math.sqrt((yaw - prev_pose[0])**2 + (pitch - prev_pose[1])**2)
        
        return velocity
    
    def _update_pattern_detection(self, yaw, pitch, current_time):
        """Update pattern detection counters"""
        # Count specific patterns
        if pitch > self.config.DOWNWARD_GAZE_ANGLE:
            self.repeated_patterns['downward_looks'] += 1
        
        if abs(yaw) > self.config.SIDE_GAZE_ANGLE:
            self.repeated_patterns['side_looks'] += 1
        
        if abs(yaw) > self.config.HEAD_MOVEMENT_THRESHOLD:
            self.repeated_patterns['head_turns'] += 1
    
    def _calculate_suspicion_increase(self, is_looking_away, duration, head_velocity, yaw, pitch):
        """Calculate how much to increase suspicion score"""
        increase = 0
        
        # Base increase for looking away
        if is_looking_away:
            increase += self.config.LOOKING_AWAY_PENALTY
            
            # Extra points for looking down (reading notes)
            if pitch > self.config.DOWNWARD_GAZE_ANGLE:
                increase += self.config.DOWNWARD_LOOK_PENALTY
            
            # Extra points for looking far to sides
            if abs(yaw) > self.config.SIDE_GAZE_ANGLE * 1.5:
                increase += self.config.SIDE_LOOK_PENALTY
        
        # Points for rapid head movements
        if head_velocity > self.config.HEAD_MOVEMENT_THRESHOLD:
            increase += self.config.HEAD_MOVEMENT_PENALTY
        
        # Duration multiplier
        if duration > self.config.SUSPICIOUS_GAZE_THRESHOLD:
            increase *= self.config.DURATION_MULTIPLIER
        
        # Decrease suspicion when looking at screen
        if not is_looking_away:
            increase = self.config.RECOVERY_RATE  # Slowly decrease suspicion
        
        return increase
    
    def _determine_alert_level(self, looking_away_duration, current_time):
        """Determine the appropriate alert level"""
        # Cooldown check
        if current_time - self.last_alert_time < self.config.ALERT_COOLDOWN:
            if self.alert_active:
                return "ONGOING_ALERT"
            return "NO_ALERT"
        
        # High suspicion score always triggers alert
        if self.suspicion_score > self.config.HIGH_SUSPICION_THRESHOLD:
            self._trigger_alert("HIGH_SUSPICION", current_time)
            return "CHEATING_DETECTED"
        
        # Duration-based alerts
        if looking_away_duration > self.config.CHEATING_GAZE_THRESHOLD:
            self._trigger_alert("PROLONGED_DISTRACTION", current_time)
            return "CHEATING_DETECTED"
        elif looking_away_duration > self.config.SUSPICIOUS_GAZE_THRESHOLD:
            return "SUSPICIOUS_BEHAVIOR"
        
        # Pattern-based alerts
        if (self.repeated_patterns['downward_looks'] > self.config.MAX_DOWNWARD_LOOKS or 
            self.repeated_patterns['side_looks'] > self.config.MAX_SIDE_LOOKS or
            self.repeated_patterns['head_turns'] > self.config.MAX_HEAD_TURNS):
            self._trigger_alert("REPETITIVE_PATTERNS", current_time)
            return "CHEATING_DETECTED"
        
        self.alert_active = False
        return "NO_ALERT"
    
    def _trigger_alert(self, reason, current_time):
        """Trigger a cheating alert"""
        self.last_alert_time = current_time
        self.alert_active = True
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'suspicion_score': self.suspicion_score,
            'total_away_time': self.total_looking_away_time,
            'patterns': self.repeated_patterns.copy(),
            'alert_info': ALERT_MESSAGES.get(reason, {'message': reason, 'severity': 'MEDIUM'})
        }
        
        self.cheating_alerts.append(alert)
        
        # Log to console
        alert_info = ALERT_MESSAGES.get(reason, {'message': reason, 'severity': 'MEDIUM'})
        print(f"\n🚨 CHEATING ALERT: {alert_info['message']}")
        print(f"   Severity: {alert_info['severity']}")
        print(f"   Suspicion Score: {self.suspicion_score}/{self.config.MAX_SUSPICION_SCORE}")
        print(f"   Total time looking away: {self.total_looking_away_time:.1f}s")
        
    def _create_detection_result(self, is_looking_away, alert_level, timestamp, extra_data=None):
        """Create standardized detection result"""
        result = {
            'timestamp': timestamp,
            'is_looking_away': is_looking_away,
            'alert_level': alert_level,
            'suspicion_score': self.suspicion_score,
            'total_alerts': len(self.cheating_alerts)
        }
        
        if extra_data:
            result.update(extra_data)
            
        return result
    
    def get_session_summary(self):
        """Get summary of the detection session"""
        session_duration = time.time() - self.session_start_time
        
        return {
            'session_duration': session_duration,
            'total_alerts': len(self.cheating_alerts),
            'total_looking_away_time': self.total_looking_away_time,
            'away_time_percentage': (self.total_looking_away_time / session_duration) * 100 if session_duration > 0 else 0,
            'final_suspicion_score': self.suspicion_score,
            'repeated_patterns': self.repeated_patterns,
            'alerts': self.cheating_alerts,
            'behavior_analysis': self._generate_behavior_analysis()
        }
    
    def _generate_behavior_analysis(self):
        """Generate detailed behavior analysis"""
        analysis = {}
        
        for pattern_type, count in self.repeated_patterns.items():
            if count > 0:
                analysis[pattern_type] = {
                    'count': count,
                    'description': BEHAVIOR_DESCRIPTIONS.get(pattern_type, 'Unknown behavior'),
                    'severity': 'HIGH' if count > 10 else 'MEDIUM' if count > 5 else 'LOW'
                }
        
        return analysis
    
    def reset_patterns(self):
        """Reset pattern counters (call periodically)"""
        for key in self.repeated_patterns:
            self.repeated_patterns[key] = max(0, self.repeated_patterns[key] - 1)
    
    def set_detection_mode(self, mode='moderate'):
        """Set detection sensitivity mode"""
        if mode == 'strict':
            CheatingDetectionConfig.set_strict_mode()
        elif mode == 'lenient':
            CheatingDetectionConfig.set_lenient_mode()
        else:
            CheatingDetectionConfig.set_moderate_mode()
        
        print(f"Detection mode set to: {mode.upper()}")
        
        