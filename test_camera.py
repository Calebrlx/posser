import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open webcam")
    exit()

ret, frame = cap.read()
print("Captured frame shape:", frame.shape if ret else "Capture failed")
cap.release()