# schemas.py
from pydantic import BaseModel
from typing import List, Dict, Union

class TextInput(BaseModel):
    text: str

class PredictionResult(BaseModel):
    label: str
    score: float

class APIResponse(BaseModel):
    text: str
    predictions: List[PredictionResult]