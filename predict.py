import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ---------------- SETTINGS ----------------
MODEL_PATH = "Best_Cattle_Breed.h5"
IMAGE_PATH = r"C:\Users\User\Downloads\WhatsApp Image 2025-12-05 at 5.50.14 PM.jpeg"
IMAGE_SIZE = (224, 224)

DATA_DIR = r"C:\Users\User\Downloads\archive\Indian_bovine_breeds\Indian_bovine_breeds"

# ---------------- LOAD MODEL ----------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# ---------------- AUTO LOAD CLASS NAMES ----------------
# class order must match training dataset alphabetical order
CLASS_NAMES = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])

print("Classes found:", CLASS_NAMES)

# ---------------- PREDICT FUNCTION ----------------
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)

    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    index = np.argmax(preds)
    confidence = float(np.max(preds))

    return CLASS_NAMES[index], confidence

# ---------------- RUN AUTOMATICALLY ----------------
breed, conf = predict_image(IMAGE_PATH)

print("\n===== PREDICTION RESULT =====")
print(f"🐄 Breed: {breed}")
print(f"🔍 Confidence: {conf:.4f}")
print("================================\n")
