import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime
import time

model_path = "bodypix_mobilenet_float_075_224.onnx"
session = ort.InferenceSession(model_path)

def preprocess(frame):
    resized = cv2.resize(frame, (224, 224))
    normalized = resized.astype(np.float32) / 127.5 - 1.0
    input_tensor = np.transpose(normalized, (2, 0, 1))  # HWC → CHW
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Batch dim
    return input_tensor

def postprocess(output, original_shape):
    mask = output[0][0]
    mask = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.uint8)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not found.")
    exit()

last_log_time = None
cooldown = 5  # seconds between logs

with open("hair_touch_log.csv", "a") as log_file:
    log_file.write("timestamp,event\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        input_tensor = preprocess(frame)
        outputs = session.run(None, {"input": input_tensor})
        mask = postprocess(outputs, frame.shape)

        # These class IDs may vary based on the model's training
        hand_mask = (mask == 15).astype(np.uint8)
        hair_mask = (mask == 1).astype(np.uint8)

        # Compute overlap (intersection)
        overlap = cv2.bitwise_and(hand_mask, hair_mask)
        overlap_area = np.sum(overlap)

        if overlap_area > 200:  # Threshold to filter false positives
            now = datetime.now()
            if not last_log_time or (now - last_log_time).total_seconds() > cooldown:
                last_log_time = now
                print(f"[{now.isoformat()}] Touch Detected")
                log_file.write(f"{now.isoformat()},touch_detected\n")
                log_file.flush()

        # Run at ~0.5 fps
        time.sleep(2)

cap.release()