import os
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Paths
DATA_DIR = r"C:\Users\mehar\Downloads\archive (29)\UTKFace"   # <-- adjust this path
MODEL_PATH = os.path.join("models", "resnet_age_model.h5")
IMG_SIZE = (224, 224)

# Parse age from filename
def parse_age(fname):
    return int(fname.split("_")[0])

# Load dataset
def load_data(data_dir, sample_limit=None):
    X, y = [], []
    detector = MTCNN()
    files = os.listdir(data_dir)
    if sample_limit:
        files = files[:sample_limit]
    for fname in files:
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        try:
            path = os.path.join(data_dir, fname)
            img = Image.open(path).convert("RGB")
            arr = np.array(img)
            results = detector.detect_faces(arr)
            if results:
                x, y0, w, h = results[0]['box']
                x, y0 = max(0, x), max(0, y0)
                face = arr[y0:y0+h, x:x+w]
                face = Image.fromarray(face).resize(IMG_SIZE)
                X.append(np.array(face))
                y.append(parse_age(fname))
        except Exception as e:
            print(f"Skipping {fname}: {e}")
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")

print("[INFO] Loading data...")
X, y = load_data(DATA_DIR, sample_limit=5000)
X = X / 255.0
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("[INFO] Building model...")
base_model = ResNet50(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
predictions = Dense(1, activation="linear")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer="adam", loss="mae", metrics=["mae"])
model.summary()

checkpoint = ModelCheckpoint(MODEL_PATH, monitor="val_mae", save_best_only=True)
earlystop = EarlyStopping(monitor="val_mae", patience=5, restore_best_weights=True)

print("[INFO] Training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[checkpoint, earlystop]
)

print(f"[INFO] Model saved to {MODEL_PATH}")
