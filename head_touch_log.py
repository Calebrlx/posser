import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime, timedelta

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
cap = cv2.VideoCapture(0)
last_touch_time = None
cooldown_seconds = 2000

# def is_touching_head(landmarks):
#     wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
#     nose = landmarks[mp_pose.PoseLandmark.NOSE]

#     # Simple Euclidean distance (values are normalized 0-1)
#     dist = np.sqrt((wrist.x - nose.x)**2 + (wrist.y - nose.y)**2 + (wrist.z - nose.z)**2)

#     return dist < 0.1  # Tune threshold based on test

def is_touching_head(landmarks):
    wrist_r = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    wrist_l = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]

    # Euclidean distances
    dist_r = np.sqrt((wrist_r.x - nose.x)**2 + (wrist_r.y - nose.y)**2 + (wrist_r.z - nose.z)**2)
    dist_l = np.sqrt((wrist_l.x - nose.x)**2 + (wrist_l.y - nose.y)**2 + (wrist_l.z - nose.z)**2)
    print(f"Distance Right: {dist_r:.3f}, Distance Left: {dist_l:.3f}")

    return dist_r < 0.1 or dist_l < 0.1 

# with open("head_touch_log.csv", "a") as log_file:
#     log_file.write("timestamp,event\n")  # Header if new

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Camera read failed.")
#             break

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(rgb)

#         if results.pose_landmarks and is_touching_head(results.pose_landmarks.landmark):
#             timestamp = datetime.now().isoformat()
#             print(f"[{timestamp}] Touch Detected")

#             log_file.write(f"{timestamp},touch_detected\n")
#             log_file.flush()


with open("head_touch_log.csv", "a") as log_file:
    log_file.write("timestamp,event\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks and is_touching_head(results.pose_landmarks.landmark):
            now = datetime.now()
            if not last_touch_time or (now - last_touch_time).total_seconds() > cooldown_seconds:
                last_touch_time = now
                print(f"[{now.isoformat()}] Touch Detected")
                log_file.write(f"{now.isoformat()},touch_detected\n")
                log_file.flush()

cap.release()