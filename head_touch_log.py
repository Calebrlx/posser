import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
cap = cv2.VideoCapture(0)

def is_touching_head(landmarks):
    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]

    # Simple Euclidean distance (values are normalized 0-1)
    dist = np.sqrt((wrist.x - nose.x)**2 + (wrist.y - nose.y)**2 + (wrist.z - nose.z)**2)
    return dist < 0.1  # Tune threshold based on test

with open("head_touch_log.csv", "a") as log_file:
    log_file.write("timestamp,event\n")  # Header if new

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks and is_touching_head(results.pose_landmarks.landmark):
            timestamp = datetime.now().isoformat()
            print(f"[{timestamp}] Touch Detected")
            log_file.write(f"{timestamp},touch_detected\n")
            log_file.flush()

cap.release()