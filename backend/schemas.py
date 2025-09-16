# backend/app/schemas/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class PredictionResponse(BaseModel):
    predicted_word: str
    confidence: float
    status: str

class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool

class PredictionRequest(BaseModel):
    image: str  # base64 encoded image