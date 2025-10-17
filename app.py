
from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import tensorflow as tf
import os

# Initialize Flask app
app = Flask(__name__)

# Folder to store uploads
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
MODEL_PATH = os.path.join("models", "resnet_age_model.h5")
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = (224, 224)

# Initialize camera (global variable)
camera = None


# ---------- IMAGE PREPROCESSING ----------
def preprocess_image(frame):
    """Resize, normalize, and prepare frame for model prediction."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_array = np.expand_dims(img_resized / 255.0, axis=0)
    return img_array


# ---------- FRAME GENERATOR ----------
def generate_frames():
    """Continuously capture frames from webcam and stream to browser."""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows

    if not camera.isOpened():
        print("❌ ERROR: Could not open webcam.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Preprocess frame and predict age
        img_array = preprocess_image(frame)
        prediction = model.predict(img_array, verbose=0)[0][0]
        predicted_age = int(prediction)

        # Draw age prediction on frame
        cv2.putText(frame, f"Age: {predicted_age}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame as part of HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ---------- FLASK ROUTES ----------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/live')
def live():
    return render_template('live.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------- MAIN ----------
if __name__ == "__main__":
    app.run(debug=True)

