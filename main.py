from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.models import load_model

# -----------------------------
# APP INIT
# -----------------------------
app = FastAPI(title="Deepfake Detection API 🚀")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# MODEL LOADING (SAFE)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "deepfake_model.h5")

print("🔍 Model path:", MODEL_PATH)
print("📁 File exists:", os.path.exists(MODEL_PATH))

model = None

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", str(e))

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
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
        if model is None:
            return {"error": "Model not loaded"}

        contents = await file.read()
        image = preprocess_image(contents)

        prediction = model.predict(image)[0][0]

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