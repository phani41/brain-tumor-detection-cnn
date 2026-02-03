from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

# --------------------------------
# APP SETUP
# --------------------------------
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MOBILENET_PATH = os.path.join(BASE_DIR, "model", "brain_tumor_model.keras")
EFFICIENTNET_PATH = os.path.join(BASE_DIR, "model", "brain_tumor_efficientnet.keras")

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
INVALID_THRESHOLD = 0.45  # confidence below this = invalid image

# --------------------------------
# LOAD MODELS
# --------------------------------
print("🧠 Loading models...")

mobilenet_model = tf.keras.models.load_model(MOBILENET_PATH)
efficientnet_model = tf.keras.models.load_model(EFFICIENTNET_PATH)

print("✅ Models loaded successfully")

# --------------------------------
# IMAGE VALIDATION
# --------------------------------
def is_valid_image(file):
    try:
        img = Image.open(file)
        img.verify()
        file.seek(0)
        return True
    except:
        return False

# --------------------------------
# PREPROCESSING
# --------------------------------
def preprocess_mobilenet(file):
    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_efficientnet(file):
    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img)
    img = eff_preprocess(img)
    return np.expand_dims(img, axis=0)

# --------------------------------
# HEALTH CHECK
# --------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Brain Tumor API running"}), 200

# --------------------------------
# COMPARE MODELS
# --------------------------------
@app.route("/compare", methods=["POST"])
def compare_models():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]

    if not is_valid_image(image):
        return jsonify({"error": "Invalid image file"}), 400

    image.seek(0)
    img_mob = preprocess_mobilenet(image)

    image.seek(0)
    img_eff = preprocess_efficientnet(image)

    mob_preds = mobilenet_model.predict(img_mob)[0]
    eff_preds = efficientnet_model.predict(img_eff)[0]

    mob_conf = float(np.max(mob_preds))
    eff_conf = float(np.max(eff_preds))

    # INVALID IMAGE CHECK
    if mob_conf < INVALID_THRESHOLD and eff_conf < INVALID_THRESHOLD:
        return jsonify({
            "invalid": True,
            "message": "Uploaded image is not a valid brain MRI"
        }), 200

    mob_idx = int(np.argmax(mob_preds))
    eff_idx = int(np.argmax(eff_preds))

    # BEST MODEL DECISION
    best_model = "mobilenet" if mob_conf >= eff_conf else "efficientnet"

    response = {
        "invalid": False,
        "best_model": best_model,
        "mobilenet": {
            "prediction": CLASSES[mob_idx],
            "confidence": round(mob_conf * 100, 2),
            "probabilities": {
                CLASSES[i]: round(float(mob_preds[i]) * 100, 2)
                for i in range(len(CLASSES))
            }
        },
        "efficientnet": {
            "prediction": CLASSES[eff_idx],
            "confidence": round(eff_conf * 100, 2),
            "probabilities": {
                CLASSES[i]: round(float(eff_preds[i]) * 100, 2)
                for i in range(len(CLASSES))
            }
        }
    }

    return jsonify(response), 200

# --------------------------------
# RUN SERVER
# --------------------------------
if __name__ == "__main__":
    app.run(debug=True)
