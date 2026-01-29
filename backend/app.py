from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# ---------------------------------
# App setup
# ---------------------------------
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MOBILENET_PATH = os.path.join(BASE_DIR, "model", "brain_tumor_model.keras")
EFFICIENTNET_PATH = os.path.join(BASE_DIR, "model", "brain_tumor_efficientnet.keras")

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# ---------------------------------
# Load models
# ---------------------------------
print("🧠 Loading models...")

mobilenet_model = tf.keras.models.load_model(MOBILENET_PATH)
efficientnet_model = tf.keras.models.load_model(EFFICIENTNET_PATH)

print("✅ Models loaded successfully")

# ---------------------------------
# Image preprocessing
# ---------------------------------
def preprocess_image(file):
    try:
        img = Image.open(file).convert("RGB")
    except:
        return None, None

    img = img.resize((224, 224))
    img_array = np.array(img)

    mobilenet_img = mobilenet_preprocess(img_array.copy())
    efficientnet_img = efficientnet_preprocess(img_array.copy())

    mobilenet_img = np.expand_dims(mobilenet_img, axis=0)
    efficientnet_img = np.expand_dims(efficientnet_img, axis=0)

    return mobilenet_img, efficientnet_img

# ---------------------------------
# Health check
# ---------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Brain Tumor API running"}), 200

# ---------------------------------
# MobileNet prediction
# ---------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    mob_img, _ = preprocess_image(request.files["image"])
    if mob_img is None:
        return jsonify({"error": "Invalid image file"}), 400

    preds = mobilenet_model.predict(mob_img)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx]) * 100

    return jsonify({
        "model": "MobileNetV2",
        "prediction": CLASSES[idx],
        "confidence": round(conf, 2),
        "probabilities": {
            CLASSES[i]: round(float(preds[i]) * 100, 2)
            for i in range(len(CLASSES))
        }
    })

# ---------------------------------
# Compare models
# ---------------------------------
@app.route("/compare", methods=["POST"])
def compare_models():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    mob_img, eff_img = preprocess_image(request.files["image"])
    if mob_img is None or eff_img is None:
        return jsonify({"error": "Invalid image file"}), 400

    mob_preds = mobilenet_model.predict(mob_img)[0]
    eff_preds = efficientnet_model.predict(eff_img)[0]

    mob_idx = int(np.argmax(mob_preds))
    eff_idx = int(np.argmax(eff_preds))

    mob_conf = float(mob_preds[mob_idx]) * 100
    eff_conf = float(eff_preds[eff_idx]) * 100

    # ---------------------------------
    # Best model decision
    # ---------------------------------
    if mob_conf >= eff_conf and mob_conf >= 50:
        best_model = "MobileNetV2"
        best_prediction = CLASSES[mob_idx]
    elif eff_conf >= 50:
        best_model = "EfficientNet"
        best_prediction = CLASSES[eff_idx]
    else:
        best_model = "Uncertain"
        best_prediction = "Low confidence – image may not be MRI"

    return jsonify({
        "mobilenet": {
            "prediction": CLASSES[mob_idx],
            "confidence": round(mob_conf, 2),
            "probabilities": {
                CLASSES[i]: round(float(mob_preds[i]) * 100, 2)
                for i in range(len(CLASSES))
            }
        },
        "efficientnet": {
            "prediction": CLASSES[eff_idx],
            "confidence": round(eff_conf, 2),
            "probabilities": {
                CLASSES[i]: round(float(eff_preds[i]) * 100, 2)
                for i in range(len(CLASSES))
            },
            "warning": (
                "Low confidence – EfficientNet unreliable"
                if eff_conf < 40 else "OK"
            )
        },
        "best_model": {
            "model": best_model,
            "prediction": best_prediction
        }
    })

# ---------------------------------
# Run server
# ---------------------------------
if __name__ == "__main__":
    app.run(debug=True)
