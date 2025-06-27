import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load model
model = load_model("model/mask_detector_model.h5")

# Labels and colors
labels = ["Mask", "No Mask"]
colors = [(0, 255, 0), (0, 0, 255)]

# Image path
image_path = "facemask.jpg"
if not os.path.exists(image_path):
    print(f"❌ File not found: {image_path}")
    exit()

image = cv2.imread(image_path)
if image is None:
    print("❌ Failed to load image.")
    exit()

orig = image.copy()

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("❌ Error loading Haar cascade.")
    exit()

# Convert to grayscale and detect faces
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# Detect and draw
if len(faces) == 0:
    print("⚠️ No faces detected.")
else:
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        pred = model.predict(face)[0][0]
        label = labels[int(pred > 0.5)]
        confidence = pred if pred > 0.5 else 1 - pred
        color = colors[int(pred > 0.5)]

        cv2.putText(orig, f"{label} ({confidence*100:.2f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(orig, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Result", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

