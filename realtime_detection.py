import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model
model_path = 'model/mask_detector_model.h5'
try:
    model = load_model(model_path)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Load Haar cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print("❌ Failed to load Haar Cascade XML file.")
    exit()

# Label and color mapping
labels = ['Mask', 'No Mask']
colors = [(0, 255, 0), (0, 0, 255)]

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("🎥 Starting real-time mask detection. Press 'q' to quit.")

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) == 0:
        cv2.putText(frame, "No Face Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face, verbose=0)[0][0]
        label_idx = int(prediction > 0.5)
        label = labels[label_idx]
        color = colors[label_idx]
        confidence = prediction if prediction > 0.5 else 1 - prediction

        cv2.putText(frame, f"{label}: {confidence*100:.2f}%", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Calculate and display FPS
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("🟢 Real-Time Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Stopping...")
        break

cap.release()
cv2.destroyAllWindows()
