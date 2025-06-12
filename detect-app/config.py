from dotenv import load_dotenv

load_dotenv()

# Cheating Detection Configuration
# Adjust these parameters based on your requirements and testing

class CheatingDetectionConfig:
    """Configuration parameters for cheating detection system"""
    
    # Time thresholds (in seconds)
    SUSPICIOUS_GAZE_THRESHOLD = 3.0    # Time looking away to trigger warning
    CHEATING_GAZE_THRESHOLD = 5.0      # Time looking away to trigger cheating alert
    ALERT_COOLDOWN = 10                # Seconds between repeated alerts
    
    # Angle thresholds (in radians)
    DOWNWARD_GAZE_ANGLE = 0.3          # ~17 degrees - looking down at notes
    SIDE_GAZE_ANGLE = 0.5              # ~29 degrees - looking at other screens
    HEAD_MOVEMENT_THRESHOLD = 0.4       # Significant head movement threshold
    
    # Pattern detection
    REPEATED_MOVEMENT_COUNT = 3         # Number of repeated movements to flag
    MAX_DOWNWARD_LOOKS = 10            # Maximum downward looks before alert
    MAX_SIDE_LOOKS = 8                 # Maximum side looks before alert  
    MAX_HEAD_TURNS = 15                # Maximum head turns before alert
    
    # Scoring system
    MAX_SUSPICION_SCORE = 100          # Maximum suspicion score
    HIGH_SUSPICION_THRESHOLD = 70      # Score threshold for high suspicion
    
    # Suspicion score increments
    LOOKING_AWAY_PENALTY = 0.5         # Base penalty for looking away
    DOWNWARD_LOOK_PENALTY = 1.0        # Extra penalty for looking down
    SIDE_LOOK_PENALTY = 1.5            # Extra penalty for side looks
    HEAD_MOVEMENT_PENALTY = 2.0        # Penalty for rapid head movements
    DURATION_MULTIPLIER = 2.0          # Multiplier for prolonged behavior
    RECOVERY_RATE = -0.2               # Score decrease when looking at screen
    
    # Detection sensitivity levels
    @classmethod
    def set_strict_mode(cls):
        """Set strict detection parameters for high-stakes exams"""
        cls.SUSPICIOUS_GAZE_THRESHOLD = 2.0
        cls.CHEATING_GAZE_THRESHOLD = 3.0
        cls.MAX_DOWNWARD_LOOKS = 5
        cls.MAX_SIDE_LOOKS = 4
        cls.HIGH_SUSPICION_THRESHOLD = 50
        
    @classmethod
    def set_moderate_mode(cls):
        """Set moderate detection parameters for interviews"""
        cls.SUSPICIOUS_GAZE_THRESHOLD = 3.0
        cls.CHEATING_GAZE_THRESHOLD = 5.0
        cls.MAX_DOWNWARD_LOOKS = 10
        cls.MAX_SIDE_LOOKS = 8
        cls.HIGH_SUSPICION_THRESHOLD = 70
        
    @classmethod
    def set_lenient_mode(cls):
        """Set lenient detection parameters for casual monitoring"""
        cls.SUSPICIOUS_GAZE_THRESHOLD = 5.0
        cls.CHEATING_GAZE_THRESHOLD = 8.0
        cls.MAX_DOWNWARD_LOOKS = 15
        cls.MAX_SIDE_LOOKS = 12
        cls.HIGH_SUSPICION_THRESHOLD = 80


# Alert messages and descriptions
ALERT_MESSAGES = {
    'PROLONGED_DISTRACTION': {
        'message': 'Candidate looking away from screen for extended period',
        'severity': 'HIGH',
        'description': 'May indicate reading notes or consulting external resources'
    },
    'HIGH_SUSPICION': {
        'message': 'High suspicion score reached',
        'severity': 'HIGH',
        'description': 'Multiple suspicious behaviors detected'
    },
    'REPETITIVE_PATTERNS': {
        'message': 'Repetitive suspicious movement patterns detected',
        'severity': 'MEDIUM',
        'description': 'Consistent looking away patterns may indicate systematic cheating'
    },
    'RAPID_HEAD_MOVEMENT': {
        'message': 'Unusual head movement patterns',
        'severity': 'MEDIUM', 
        'description': 'Rapid or repetitive head movements may indicate communication with others'
    }
}

# Behavior descriptions for reporting
BEHAVIOR_DESCRIPTIONS = {
    'downward_looks': 'Looking down (potentially reading notes or using phone)',
    'side_looks': 'Looking to the side (potentially viewing other screens or people)',
    'head_turns': 'Frequent head turning (potentially communicating with others)',
    'prolonged_absence': 'Extended periods without face detection',
    'rapid_movements': 'Rapid head movements (potentially checking surroundings)'
}
