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

# -----------------------------
# Step 1: Load and Prepare Dataset (from train/test folders)
# -----------------------------

data_dir = 'dataset'

X = []
y = []
emotion_labels = []

print("📁 Scanning dataset folders...")
for subset in ['train', 'test']:
    subset_path = os.path.join(data_dir, subset)
    if not os.path.exists(subset_path):
        print(f"⚠️ Skipping missing folder: {subset_path}")
        continue

    for label in os.listdir(subset_path):
        label_path = os.path.join(subset_path, label)
        if not os.path.isdir(label_path):
            continue

        if label not in emotion_labels:
            emotion_labels.append(label)

        label_index = emotion_labels.index(label)

        img_files = os.listdir(label_path)
        print(f"📦 Loading {len(img_files)} images from '{subset}/{label}'")
        for img_name in tqdm(img_files, desc=f"{subset}/{label}"):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (48, 48))
            X.append(img)
            y.append(label_index)

emotion_labels.sort()
print("✅ Emotion classes used:", emotion_labels)

X = np.array(X).astype('float32') / 255.0
X = np.expand_dims(X, -1)
y = to_categorical(y, num_classes=len(emotion_labels))

print(f"✅ Dataset loaded: {len(X)} total images")

# -----------------------------
# Step 2: Split Data
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"🧪 Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# -----------------------------
# Step 3: Build and Train Model
# -----------------------------

MODEL_FILENAME = "face_expression_model.h5"

if os.path.exists(MODEL_FILENAME):
    print(f"ℹ️ Model already exists. Loading '{MODEL_FILENAME}'...")
    model = load_model(MODEL_FILENAME)
else:
    print("⚙️ Building and training model...")
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

    print("🚀 Training started. This may take a few minutes...")
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))
    print("✅ Training completed!")

    model.save(MODEL_FILENAME)
    print(f"💾 Model saved as '{MODEL_FILENAME}'.")

    # -----------------------------
    # Step 4: Evaluate with Confusion Matrix
    # -----------------------------

    print("📊 Generating confusion matrix and classification report...")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig("confusion_matrix.png")
    plt.show()

    print("\n🧾 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=emotion_labels))

# -----------------------------
# Step 5: Predict Emotion from Image
# -----------------------------

def predict_emotion_from_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ ERROR: Image file '{image_path}' not found!")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("❌ ERROR: Could not read the image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        print("⚠️ No face detected in the image.")
        return

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, -1)

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

# -----------------------------
# Step 6: Test with Your Image
# -----------------------------

test_image_path = "test_image.jpeg"
predict_emotion_from_image(test_image_path)
