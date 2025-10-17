import cv2
import numpy as np
import tensorflow as tf
import os

# ---------------------------
# 1. Load trained model
# ---------------------------
MODEL_PATH = r"C:\Users\mehar\Desktop\AGE\models\resnet_age_model.h5"

print("[INFO] Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = (224, 224)

def predict_age_from_frame(frame):
    """Preprocess and predict age for a single frame"""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_array = np.expand_dims(img_resized / 255.0, axis=0)
    pred = model.predict(img_array, verbose=0)[0][0]
    return int(pred)

# ---------------------------
# 2. Ask user input
# ---------------------------
print("\nChoose an option:")
print("1. Test with image")
print("2. Live webcam detection")
choice = input("Enter choice (1/2): ").strip()

# ---------------------------
# 3. Option 1 — Image input
# ---------------------------
if choice == "1":
    image_path = input("Enter image path: ").strip()

    if not os.path.exists(image_path):
        print(f"❌ Image not found at {image_path}")
        exit()

    frame = cv2.imread(image_path)
    predicted_age = predict_age_from_frame(frame)
    print(f"✅ Predicted Age: {predicted_age} years")

    cv2.putText(frame, f"Age: {predicted_age}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Predicted Age", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------------------------
# 4. Option 2 — Live webcam detection
# ---------------------------
elif choice == "2":
    print("[INFO] Starting webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot access camera.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        predicted_age = predict_age_from_frame(frame)
        cv2.putText(frame, f"Age: {predicted_age}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Live Age Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("❌ Invalid choice. Run again and choose 1 or 2.")
