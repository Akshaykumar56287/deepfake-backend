from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from PIL import Image
import io
import os
from tensorflow.keras.models import load_model

# -----------------------------
# APP INITIALIZATION
# -----------------------------
app = FastAPI(title="Deepfake Detection API")

# Allow frontend access (important for Vite/React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# LOAD MODEL (DOCKER SAFE)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "deepfake_model.h5")

print("🔍 Model path:", MODEL_PATH)
print("📁 File exists:", os.path.exists(MODEL_PATH))

model = load_model(MODEL_PATH)

# -----------------------------
# IMAGE PREPROCESSING FUNCTION
# -----------------------------
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def home():
    return {"message": "Deepfake AI API Running 🚀"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Preprocess image
        image = preprocess_image(contents)

        # Prediction
        prediction = model.predict(image)[0][0]

        # Convert result
        if prediction > 0.5:
            result = "Fake"
            confidence = float(prediction * 100)
        else:
            result = "Real"
            confidence = float((1 - prediction) * 100)

        return {
            "prediction": result,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {"error": str(e)}