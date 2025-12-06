import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------------- SETTINGS ----------------
MODEL_PATH = "Best_Cattle_Breed.h5"
IMAGE_SIZE = (224, 224)

# ---------------- MANUAL CLASS NAMES ----------------
CLASS_NAMES = [
    "Umblachery", "Tharparkar", "Toda", "Sahiwal", "Surti",
    "Red_Dane", "Rathi", "Pulikulam", "Ongole", "Nimari",
    "Nagpuri", "Nili_Ravi", "Nagori", "Murrah", "Mehsana",
    "Malnad_Gidda", "Krishna_Valley", "Khillari", "Kasargod",
    "Kenkatha", "Kherigarh", "Kankrej", "Kangayam", "Jaffrabadi",
    "Jersey", "Holstein_Friesian", "Hariana", "Hallikar",
    "Guernsey", "Gir", "Deoni", "Dangi", "Bhadawari",
    "Brown_Swiss", "Bargur", "Banni", "Ayrshire", "Amritmahal",
    "Alambadi"
]

print("Loaded class names:", CLASS_NAMES)
print("Number of classes:", len(CLASS_NAMES))

# ---------------- FLASK APP ----------------
app = Flask(__name__)
CORS(app)  # Enable CORS for Angular, mobile, browser

# ---------------- LOAD MODEL ----------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

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
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
    return label, conf

# ======================================================
# 📌 DEBUG ENDPOINT — Shows keys Thunder/Angular sends
# ======================================================
@app.route("/debug", methods=["POST"])
def debug():
    print("FILES:", request.files)
    print("FORM:", request.form)
    return jsonify({
        "files_received": list(request.files.keys()),
        "form_received": request.form.to_dict()
    })

# ======================================================
# 📌 MAIN PREDICTION ENDPOINT
# ======================================================
@app.route("/predict", methods=["POST"])
def predict_api():

    print("FILES:", request.files)

    # Accept BOTH "image" and "file"
    file = request.files.get("image") or request.files.get("file")

    if file is None:
        return jsonify({"error": "No image provided"}), 400

    try:
        img = Image.open(file.stream)
    except Exception as e:
        print("Image error:", e)
        return jsonify({"error": "Invalid image"}), 400

    breed, confidence = predict(img)

    return jsonify({
        "breed": breed,
        "confidence": confidence
    })

# ---------------- ROOT ENDPOINT ----------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Cow Breed Prediction API is running!"})

# ---------------- RUN LOCAL ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
