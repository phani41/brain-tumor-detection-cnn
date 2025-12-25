import os
import numpy as np
import cv2
import tensorflow as tf
import gdown
from flask import Flask, request, jsonify
from flask_cors import CORS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


app = Flask(__name__)
CORS(app)

# ===============================
# MODEL DOWNLOAD CONFIG
# ===============================
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_model.keras")
MODEL_URL = "https://drive.google.com/uc?id=1Cav92Wcw8UBANF2npwP6q1VopX6ocf4Q"

# Create model directory if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download model if missing
if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("✅ Model downloaded successfully")

# Load model
print("🧠 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# Class labels (IMPORTANT: must match training order)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# ===============================
# IMAGE PREPROCESSING
# ===============================
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ===============================
# ROUTES
# ===============================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_array = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    processed = preprocess_image(image)
    predictions = model.predict(processed)[0]

    result = {
        "prediction": CLASS_NAMES[int(np.argmax(predictions))],
        "confidence": float(np.max(predictions)),
        "probabilities": {
            CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))
        }
    }

    return jsonify(result)

# ===============================
# START SERVER
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
