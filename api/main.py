import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Charger le modèle
model = joblib.load("models/best_rf_pipeline.joblib")

# Définition des données entrantes
class InputData(BaseModel):
    PropertyGFATotal: float = Field(..., description="Surface totale du bâtiment en pieds carrés")
    NumberofFloors: int = Field(..., description="Nombre d'étages")
    NumberofBuildings: int = Field(..., description="Nombre total de bâtiments")
    YearBuilt: int = Field(..., description="Année de construction")
    HasGas: bool = Field(..., description="Présence de gaz (True/False)")
    HasElectricity: bool = Field(..., description="Présence d'électricité (True/False)")
    HasSteam: bool = Field(..., description="Présence de vapeur (True/False)")
    HasParking: bool = Field(..., description="Présence de parking (True/False)")
    UsageCount: str = Field(..., description="Mono-usage = '0', Multi-usage = '1'")
    PropertyTypeGrouped: str = Field(..., description="Catégorie du type de propriété (ex : 'Autre', 'Hotel', ...)")
    PrimaryPropertyType: str = Field(..., description="Type principal du bâtiment")
    Neighborhood: str = Field(..., description="Quartier (ex : 'Downtown', 'Ballard', etc.)")

# Initialiser FastAPI
app = FastAPI(title="API de Prédiction Énergie - Seattle", description="Estime la consommation d'énergie des bâtiments non résidentiels à Seattle")

# Endpoint principal
@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])

    # Ajouter colonnes calculées
    df["Age"] = 2015 - df["YearBuilt"]
    df["IsRecent"] = df["YearBuilt"] >= 2010
    df["IsLarge"] = df["PropertyGFATotal"] > 400000

    # Prédiction
    prediction = model.predict(df)
    return {"prediction": round(float(prediction[0]), 2)}

