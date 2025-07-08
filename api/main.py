# api/main.py

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Charger le modèle entraîné
model = joblib.load("models/best_rf_pipeline.joblib")

# Définition des données attendues
class InputData(BaseModel):
    PropertyGFATotal: float
    NumberofFloors: int
    NumberofBuildings: int
    Age: int
    YearBuilt: int
    HasGas: bool
    HasElectricity: bool
    HasSteam: bool
    HasParking: bool
    IsLarge: bool
    IsRecent: bool
    UsageCount: str
    PropertyTypeGrouped: str
    PrimaryPropertyType: str
    Neighborhood: str

# Initialisation de FastAPI
app = FastAPI()

# Endpoint de prédiction
@app.post("/predict")
def predict(data: InputData):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    return {"prediction": round(float(prediction[0]), 2)}

