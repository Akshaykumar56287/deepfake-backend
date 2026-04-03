from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import io
from PIL import Image
from tensorflow.keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("../deepfake-project/deepfake_model.h5")

IMG_SIZE = 128

@app.post("/detect")
async def detect(file: UploadFile = File(...)):   # ⚠️ THIS LINE IS IMPORTANT
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = np.array(image)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.reshape(image, (1, IMG_SIZE, IMG_SIZE, 3))

    prediction = model.predict(image)

    real_prob = prediction[0][0]
    fake_prob = prediction[0][1]

    if real_prob > fake_prob:
        result = "Real"
        confidence = real_prob * 100
    else:
        result = "Fake"
        confidence = fake_prob * 100

    return {
        "prediction": result,
        "confidence": round(float(confidence), 2)
    }

@app.get("/")
def home():
    return {"message": "Deepfake AI API Running"}