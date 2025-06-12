import numpy as np
import math

class GazeCalibrator:
    """
    Gaze calibration using facial landmarks and geometric relationships
    Automatically adapts to user's facial structure
    """
    def __init__(self):
        self.face_baseline = None
        self.eye_baseline = None
        self.head_pose_baseline = None
        self.adaptation_frames = 30  # Frames to establish baseline
        self.frame_count = 0
        self.calibration_data = {
            'eye_distances': [],
            'face_dimensions': [],
            'head_poses': [],
            'pupil_positions': []
        }
        
    def analyze_facial_structure(self, face_data):
        """Analyze facial structure to establish personal baselines"""
        landmarks = face_data.get("landmarks", [])
        if len(landmarks) < 5:  # Need at least 5 landmarks
            return None
            
        # Extract key facial measurements
        face_width = face_data["width"]
        face_height = face_data["height"]
        
        # Calculate eye positions and distances
        left_eye, right_eye = None, None
        nose_tip = None
        
        # Find eye and nose landmarks (assuming standard landmark order)
        for i, landmark in enumerate(landmarks):
            if i == 0:  # Left eye (approximate)
                left_eye = (landmark["x"], landmark["y"])
            elif i == 1:  # Right eye (approximate)
                right_eye = (landmark["x"], landmark["y"])
            elif i == 2:  # Nose tip (approximate)
                nose_tip = (landmark["x"], landmark["y"])
                
        if left_eye and right_eye:
            eye_distance = math.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
            eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
            
            return {
                'face_dimensions': (face_width, face_height),
                'eye_distance': eye_distance,
                'eye_center': eye_center,
                'left_eye': left_eye,
                'right_eye': right_eye,
                'nose_tip': nose_tip
            }
        return None
    
    def update_baseline(self, gaze_data, facial_structure):
        """Update baseline measurements for calibration"""
        if facial_structure is None:
            return False
            
        self.calibration_data['face_dimensions'].append(facial_structure['face_dimensions'])
        self.calibration_data['eye_distances'].append(facial_structure['eye_distance'])
        
        # Store head pose
        yaw, pitch = gaze_data.get('yaw', 0), gaze_data.get('pitch', 0)
        self.calibration_data['head_poses'].append((yaw, pitch))
        
        self.frame_count += 1
        
        # After enough frames, establish baseline
        if self.frame_count >= self.adaptation_frames:
            self._establish_baseline()
            return True
        return False
    
    def _establish_baseline(self):
        """Establish baseline measurements from collected data"""
        if len(self.calibration_data['face_dimensions']) > 0:
            # Calculate average face dimensions
            face_dims = np.array(self.calibration_data['face_dimensions'])
            self.face_baseline = {
                'width': np.mean(face_dims[:, 0]),
                'height': np.mean(face_dims[:, 1]),
                'width_std': np.std(face_dims[:, 0]),
                'height_std': np.std(face_dims[:, 1])
            }
            
            # Calculate average eye distance
            self.eye_baseline = {
                'distance': np.mean(self.calibration_data['eye_distances']),
                'distance_std': np.std(self.calibration_data['eye_distances'])
            }
            
            # Calculate head pose baseline (neutral position)
            head_poses = np.array(self.calibration_data['head_poses'])
            self.head_pose_baseline = {
                'yaw': np.mean(head_poses[:, 0]),
                'pitch': np.mean(head_poses[:, 1]),
                'yaw_std': np.std(head_poses[:, 0]),
                'pitch_std': np.std(head_poses[:, 1])
            }
    
    def calibrated_gaze_prediction(self, gaze_data, facial_structure):
        """Predict calibrated gaze"""
        if not self.is_calibrated():
            return None
            
        # Get raw gaze angles
        raw_yaw = gaze_data.get('yaw', 0)
        raw_pitch = gaze_data.get('pitch', 0)
        
        # Compensate for head pose deviation from baseline
        head_yaw_offset = raw_yaw - self.head_pose_baseline['yaw']
        head_pitch_offset = raw_pitch - self.head_pose_baseline['pitch']
        
        # Scale based on facial structure differences
        face_scale_x = facial_structure['face_dimensions'][0] / self.face_baseline['width']
        face_scale_y = facial_structure['face_dimensions'][1] / self.face_baseline['height']
        
        # Apply calibration adjustments
        calibrated_yaw = head_yaw_offset * face_scale_x
        calibrated_pitch = head_pitch_offset * face_scale_y
        
        # Apply smoothing and boundary constraints
        calibrated_yaw = np.clip(calibrated_yaw, -np.pi/2, np.pi/2)
        calibrated_pitch = np.clip(calibrated_pitch, -np.pi/3, np.pi/3)
        
        return {
            'calibrated_yaw': calibrated_yaw,
            'calibrated_pitch': calibrated_pitch,
            'raw_yaw': raw_yaw,
            'raw_pitch': raw_pitch,
            'confidence': self._calculate_confidence(facial_structure)
        }
    
    def _calculate_confidence(self, facial_structure):
        """Calculate confidence score for gaze prediction"""
        if not self.face_baseline or not facial_structure:
            return 0.0
            
        # Face size consistency
        face_width_diff = abs(facial_structure['face_dimensions'][0] - self.face_baseline['width'])
        face_height_diff = abs(facial_structure['face_dimensions'][1] - self.face_baseline['height'])
        
        size_consistency = 1.0 - min(1.0, (face_width_diff + face_height_diff) / 200)
        
        # Eye distance consistency
        eye_dist_diff = abs(facial_structure['eye_distance'] - self.eye_baseline['distance'])
        eye_consistency = 1.0 - min(1.0, eye_dist_diff / 50)
        
        return (size_consistency + eye_consistency) / 2
    
    def is_calibrated(self):
        """Check if calibrator has enough data for predictions"""
        return (self.face_baseline is not None and 
                self.eye_baseline is not None and 
                self.head_pose_baseline is not None)