import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime
import time

MODEL_PATH = "bodypix_mobilenet_float_075_224.onnx"
INPUT_SIZE = (224, 224)
COOLDOWN_SECONDS = 5
OVERLAP_THRESHOLD = 200

# Initialize ONNX model
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# Preprocess frame for model
def preprocess(frame):
    resized = cv2.resize(frame, INPUT_SIZE)
    normalized = resized.astype(np.float32) / 127.5 - 1.0
    input_tensor = np.expand_dims(normalized, axis=0)  # Batch dimension
    return input_tensor

# Postprocess model output to get segmentation mask (from heatmap or segments)
def postprocess(outputs, original_shape):
    segments = outputs[0]  # float_segments:0
    mask = segments[0, :, :, 0]  # Use first (and only) channel of float_segments
    mask_resized = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask_resized.astype(np.uint8)

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not found.")
    exit()

last_log_time = None

with open("hair_touch_log.csv", "a") as log_file:
    log_file.write("timestamp,event\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed.")
            break

        input_tensor = preprocess(frame)
        outputs = session.run(None, {input_name: input_tensor})
        mask = postprocess(outputs, frame.shape)

        # Debug: print unique class labels in the segmentation output
        unique_labels = np.unique(mask)
        print(f"[DEBUG] Unique labels in mask: {unique_labels}")

        # Assumption: label IDs must be identified empirically
        HAND_CLASS_ID = 15  # Placeholder - may need adjustment
        HAIR_CLASS_ID = 1   # Placeholder - may need adjustment

        hand_mask = (mask == HAND_CLASS_ID).astype(np.uint8)
        hair_mask = (mask == HAIR_CLASS_ID).astype(np.uint8)

        overlap = cv2.bitwise_and(hand_mask, hair_mask)
        overlap_area = np.sum(overlap)

        print(f"[DEBUG] Overlap area: {overlap_area}")

        if overlap_area > OVERLAP_THRESHOLD:
            now = datetime.now()
            if last_log_time is None or (now - last_log_time).total_seconds() >= COOLDOWN_SECONDS:
                last_log_time = now
                log_entry = f"{now.isoformat()},touch_detected\n"
                print(log_entry.strip())
                log_file.write(log_entry)
                log_file.flush()

        time.sleep(2)  # Throttle to ~0.5 FPS

cap.release()