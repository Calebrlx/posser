import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "bodypix_mobilenet_float_075_224.onnx"
INPUT_SIZE = (224, 224)

# Colors (BGR)
COLORS = {
    "hand": (0, 255, 255),  # Yellow
    "hair": (255, 0, 255),  # Magenta
}

# Adjust class IDs after observing model output
HAND_CLASS_ID = 15  # Placeholder — may need to update
HAIR_CLASS_ID = 1   # Placeholder — may need to update

# Initialize ONNX session
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

def preprocess(frame):
    resized = cv2.resize(frame, INPUT_SIZE)
    normalized = resized.astype(np.float32) / 127.5 - 1.0
    return np.expand_dims(normalized, axis=0)

def postprocess(outputs, shape):
    mask = outputs[0][0, :, :, 0]
    mask_resized = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask_resized.astype(np.uint8)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not found.")
    exit()

print("[INFO] Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera frame read failed.")
        break

    input_tensor = preprocess(frame)
    outputs = session.run(None, {input_name: input_tensor})
    mask = postprocess(outputs, frame.shape)

    # Debug: unique labels
    unique = np.unique(mask)
    print(f"[DEBUG] Unique labels: {unique}")

    # Generate color overlay
    overlay = frame.copy()
    hand_mask = (mask == HAND_CLASS_ID).astype(np.uint8)
    hair_mask = (mask == HAIR_CLASS_ID).astype(np.uint8)

    overlay[hand_mask == 1] = COLORS["hand"]
    overlay[hair_mask == 1] = COLORS["hair"]

    # Blend overlay with original
    visual = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    # Display
    cv2.imshow("BodyPix Segmentation", visual)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()