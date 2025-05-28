import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import argparse

# -----------------------------
# Argument Parser for modes
# -----------------------------
parser = argparse.ArgumentParser(description="Facial Expression Recognition")
parser.add_argument('--mode', choices=['train', 'test'], required=True, help="train or test")
parser.add_argument('--image', type=str, help="Path to test image (required for test mode)")
args = parser.parse_args()

# -----------------------------
# Configuration
# -----------------------------
MODEL_FILENAME = "face_expression_model.h5"
DATASET_PATH = "dataset"
IMAGE_SIZE = (48, 48)

# Sorted emotion labels (to avoid mismatches)
emotion_labels = sorted(os.listdir(os.path.join(DATASET_PATH, "train")))

# -----------------------------
# Mode: Train
# -----------------------------
if args.mode == 'train':
    print("📁 Scanning and loading dataset...")

    X = []
    y = []

    for subset in ['train', 'test']:
        subset_path = os.path.join(DATASET_PATH, subset)
        if not os.path.exists(subset_path):
            print(f"⚠️ Skipping missing folder: {subset_path}")
            continue

        for label in os.listdir(subset_path):
            label_path = os.path.join(subset_path, label)
            if not os.path.isdir(label_path):
                continue

            label_index = emotion_labels.index(label.lower())


            img_files = os.listdir(label_path)
            print(f"📦 Loading {len(img_files)} images from '{subset}/{label}'")
            for img_name in tqdm(img_files, desc=f"{subset}/{label}"):
                img_path = os.path.join(label_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, IMAGE_SIZE)
                X.append(img)
                y.append(label_index)

    X = np.array(X).astype('float32') / 255.0
    X = np.expand_dims(X, -1)
    y = to_categorical(y, num_classes=len(emotion_labels))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"🧪 Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(emotion_labels), activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("🚀 Training started...")
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))
    print("✅ Training completed!")

    model.save(MODEL_FILENAME)
    print(f"💾 Model saved as '{MODEL_FILENAME}'.")

    print("📊 Generating confusion matrix...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig("confusion_matrix.png")
    plt.show()

    print("\n🧾 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=emotion_labels))

# -----------------------------
# Mode: Test
# -----------------------------
elif args.mode == 'test':
    if not args.image:
        print("❌ Please provide an image path with --image")
        exit()

    if not os.path.exists(MODEL_FILENAME):
        print(f"❌ ERROR: Trained model '{MODEL_FILENAME}' not found.")
        exit()

    model = load_model(MODEL_FILENAME)
    image_path = args.image

    image = cv2.imread(image_path)
    if image is None:
        print("❌ ERROR: Could not read the image.")
        exit()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        print("⚠️ No face detected in the image.")
        exit()

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, IMAGE_SIZE)
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))

        prediction = model.predict(roi)
        emotion = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        print(f"✅ Detected Emotion: {emotion} (Confidence: {confidence:.2f})")

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Emotion Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
