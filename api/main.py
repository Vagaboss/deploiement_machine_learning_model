# api/main.py

from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# Création de l'app FastAPI
app = FastAPI(
    title="API de Prédiction Énergétique des Bâtiments",
    description="API pour prédire la consommation énergétique de bâtiments non résidentiels à Seattle.",
    version="1.0.0"
)

# Chargement du modèle pré-entraîné
model = joblib.load("models/best_rf_pipeline.joblib")

# Définition du schéma des données attendues (entrée utilisateur)
class BuildingFeatures(BaseModel):
    PropertyGFATotal: float
    NumberofFloors: int
    NumberofBuildings: int
    Age: int
    HasGas: bool
    HasElectricity: bool
    PrimaryPropertyType: str
    Neighborhood: str

# Endpoint racine
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction d’énergie ! 🚀"}

# Endpoint de prédiction
@app.post("/predict")
def predict(data: BuildingFeatures):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    return {"prediction": round(prediction[0], 2)}

# Endpoint de vérification de santé
@app.get("/health")
def health_check():
    return {"status": "API opérationnelle ✅"}
