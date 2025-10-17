import cv2 
import numpy as np
import tensorflow as tf
import os

# Path to your trained model
#MODEL_PATH = os.path.join("models", "resnet_age_model.h5")
MODEL_PATH = r"C:\Users\mehar\Desktop\AGE\models\resnet_age_model.h5"

# Load the trained model
print("[INFO] Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Path to a test image (change this to any image you want to test)
TEST_IMAGE = r"C:\Users\mehar\Desktop\AGE\Picture1jcb.png"

# Check if file exists
if not os.path.exists(TEST_IMAGE):
 raise FileNotFoundError(f"❌ Test image not found at {TEST_IMAGE}")

# Load and preprocess the image
print("[INFO] Preprocessing test image...")
img = cv2.imread(TEST_IMAGE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (224, 224))  # Resize to model input
img_array = np.expand_dims(img_resized / 255.0, axis=0)  # Normalize

# Make prediction
print("[INFO] Making prediction...")
predicted_age = model.predict(img_array)[0][0]

# Show result
print(f"✅ Predicted Age: {predicted_age:.2f} years")

# (Optional) Display image with result using OpenCV
cv2.putText(img, f"Predicted Age: {int(predicted_age)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("Predicted Age", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
