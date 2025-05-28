import cv2
import numpy as np
from keras.models import load_model
import os

MODEL_FILENAME = "face_expression_model.h5"
IMAGE_SIZE = (48, 48)

# Load the trained model
if not os.path.exists(MODEL_FILENAME):
    print(f"❌ ERROR: Model file '{MODEL_FILENAME}' not found. Train the model first.")
    exit()
model = load_model(MODEL_FILENAME)

# Load emotion labels - update this path if your dataset path is different
emotion_labels = sorted(os.listdir("dataset/train"))

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)

print("📷 Starting real-time facial expression recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, IMAGE_SIZE)
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))

        prediction = model.predict(roi)
        emotion = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{emotion} ({confidence:.2f})"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
