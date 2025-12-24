from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

# Load model once
model = tf.keras.models.load_model("model/brain_tumor_model.keras")

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]

        img = cv2.imdecode(
            np.frombuffer(file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        # Preprocess (must match training)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])

        probabilities = {
            CLASS_NAMES[i]: round(float(preds[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        label = CLASS_NAMES[idx]
        if confidence < 0.7:
            label = "uncertain"

        return jsonify({
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
