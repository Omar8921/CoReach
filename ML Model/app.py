from typing import Any, Dict
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import RootModel
import uvicorn, os
import json
import joblib
import pandas as pd

app = FastAPI(title="Coral Bleaching Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load('finalModel.joblib')

@app.post("/predict")
async def predict_file(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        payload = json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON file: {e}")
    try:
        input = pd.DataFrame([payload])
        value = model.predict(input)[0]
        return abs(value * 100)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
    
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
