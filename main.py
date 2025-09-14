# backend/app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import io
from PIL import Image
import numpy as np
import json
import os

from prediction_service import PredictionService
from schemas import PredictionResponse, HealthResponse

app = FastAPI(title="Sign Language to Subtitle API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service
prediction_service = PredictionService()

@app.get("/")
async def root():
    return {"message": "Sign Language to Subtitle API"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        message="API is running",
        model_loaded=prediction_service.is_model_loaded()
    )

@app.get("/api/words")
async def get_supported_words():
    words = prediction_service.get_supported_words()
    return {"words": words}

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_sign_language(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Make prediction
        result = prediction_service.predict(image)
        
        return PredictionResponse(
            predicted_word=result["predicted_word"],
            confidence=result["confidence"],
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict-base64")
async def predict_from_base64(data: dict):
    try:
        base64_image = data.get("image")
        if not base64_image:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Remove data URL prefix if present
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Make prediction
        result = prediction_service.predict(image)
        
        return PredictionResponse(
            predicted_word=result["predicted_word"],
            confidence=result["confidence"],
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)