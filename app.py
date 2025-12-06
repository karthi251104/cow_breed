import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS   # <-- ADD THIS

# ---------------- SETTINGS ----------------
MODEL_PATH = "Best_Cattle_Breed.h5"
IMAGE_SIZE = (224, 224)
DATA_DIR = r"C:\Users\User\Downloads\archive\Indian_bovine_breeds\Indian_bovine_breeds"

# ---------------- FLASK APP ----------------
app = Flask(__name__)
CORS(app)   # <-- ENABLE CORS FOR ALL ROUTES

# ---------------- LOAD MODEL ----------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")

# ---------------- LOAD CLASS NAMES ----------------
if os.path.isdir(DATA_DIR):
    CLASS_NAMES = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])
else:
    CLASS_NAMES = []

print("Classes:", CLASS_NAMES)

# ---------------- IMAGE PREPROCESS ----------------
def preprocess(img):
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------- PREDICT FUNCTION ----------------
def predict(img):
    arr = preprocess(img)
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    label = CLASS_NAMES[idx] if CLASS_NAMES else str(idx)
    return label, conf

# ======================================================
# 📌 API: /predict  — upload an image & get JSON response
# ======================================================
@app.route("/predict", methods=["POST"])
def predict_api():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    try:
        img = Image.open(file.stream)
    except:
        return jsonify({"error": "Invalid image"}), 400

    breed, confidence = predict(img)

    return jsonify({
        "breed": breed,
        "confidence": confidence
    })

# ---------------- RUN (local only) ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
