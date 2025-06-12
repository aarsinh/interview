import numpy as np
import cv2

def draw_gaze_with_alerts(img: np.ndarray, gaze: dict, detection_result: dict):
    """Enhanced gaze drawing with cheating detection alerts"""
    # Original gaze drawing
    img = draw_gaze(img, gaze)
    
    # Get alert level and suspicion score
    alert_level = detection_result.get('alert_level', 'NO_ALERT')
    suspicion_score = detection_result.get('suspicion_score', 0)
    is_looking_away = detection_result.get('is_looking_away', False)
    
    # Draw alert indicators
    height, width = img.shape[:2]
    
    # Suspicion score bar
    bar_width = int(width * 0.3)
    bar_height = 20
    bar_x = width - bar_width - 20
    bar_y = 20
    
    # Background bar
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    
    # Fill bar based on suspicion score
    fill_width = int((suspicion_score / 100) * bar_width)
    if suspicion_score < 30:
        color = (0, 255, 0)  # Green
    elif suspicion_score < 70:
        color = (0, 255, 255)  # Yellow
    else:
        color = (0, 0, 255)  # Red
    
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
    
    # Suspicion score text
    cv2.putText(img, f"Risk: {suspicion_score:.0f}%", 
                (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Alert level indicator
    if alert_level == "CHEATING_DETECTED":
        # Red flashing border
        cv2.rectangle(img, (0, 0), (width-1, height-1), (0, 0, 255), 10)
        cv2.putText(img, "CHEATING DETECTED", 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    elif alert_level == "SUSPICIOUS_BEHAVIOR":
        # Yellow border
        cv2.rectangle(img, (0, 0), (width-1, height-1), (0, 255, 255), 5)
        cv2.putText(img, "SUSPICIOUS BEHAVIOR", 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Looking away indicator
    if is_looking_away:
        cv2.putText(img, "LOOKING AWAY", 
                    (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    
    return img


def draw_gaze(img: np.ndarray, gaze: dict):
    # draw face bounding box
    face = gaze["face"]
    x_min = int(face["x"] - face["width"] / 2)
    x_max = int(face["x"] + face["width"] / 2)
    y_min = int(face["y"] - face["height"] / 2)
    y_max = int(face["y"] + face["height"] / 2)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

    # draw gaze arrow
    _, imgW = img.shape[:2]
    arrow_length = imgW / 2
    dx = -arrow_length * np.sin(gaze["yaw"]) * np.cos(gaze["pitch"])
    dy = -arrow_length * np.sin(gaze["pitch"])
    cv2.arrowedLine(
        img,
        (int(face["x"]), int(face["y"])),
        (int(face["x"] + dx), int(face["y"] + dy)),
        (0, 0, 255),
        2,
        cv2.LINE_AA,
        tipLength=0.18,
    )

    # draw keypoints
    for keypoint in face["landmarks"]:
        color, thickness, radius = (0, 255, 0), 2, 2
        x, y = int(keypoint["x"]), int(keypoint["y"])
        cv2.circle(img, (x, y), thickness, color, radius)

    # draw label and score
    label = "yaw {:.2f}  pitch {:.2f}".format(
        gaze["yaw"] / np.pi * 180, gaze["pitch"] / np.pi * 180
    )
    cv2.putText(
        img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3
    )

    return img
