import cv2
import numpy as np

# Load Face Detection Model
FACE_PROTO = r"C:\Users\ajayh\Desktop\MINOR 2\opencv_face_detector.pbtxt"
FACE_MODEL = r"C:\Users\ajayh\Desktop\MINOR 2\opencv_face_detector_uint8.pb"

face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)

def detect_faces(frame):
    """Detects faces in an input image using OpenCV DNN model."""
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()
    return detections

# Load Sample Image (Ensure the path is correct)
IMAGE_PATH = r"C:\Users\ajayh\Desktop\MINOR 2\female_1.jpg"
image = cv2.imread(IMAGE_PATH)

# Unit Test: Check if Image is Loaded
assert image is not None, f"Error: Could not load image from {IMAGE_PATH}. Please check the file path."
print("✅ Image loaded successfully for Unit Test.")

# Perform Face Detection
detections = detect_faces(image)

# Unit Test: Check if Detections are Generated
assert detections is not None, "❌ Error: No output received from face detection model."
print("✅ Face detection model executed successfully.")

# Unit Test: Check if Any Faces Are Detected
detected_faces = 0
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.7:  # Confidence threshold
        detected_faces += 1

if detected_faces > 0:
    print(f"✅ Unit Test Passed: {detected_faces} face(s) detected successfully.")
else:
    print("❌ Unit Test Failed: No face detected. Try using a clearer image.")
