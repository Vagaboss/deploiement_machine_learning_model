import joblib
import pandas as pd
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from datetime import datetime

from api.db import SessionLocal
from api.models import Input, Output

# Charger le modèle
model = joblib.load("models/best_rf_pipeline.joblib")

# Initialiser FastAPI
app = FastAPI(
    title="API de Prédiction Énergie - Seattle",
    description="Estime la consommation d'énergie des bâtiments non résidentiels à Seattle"
)

# Dépendance pour la session DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Schéma de données entrantes
class InputData(BaseModel):
    PropertyGFATotal: float
    NumberofFloors: int
    NumberofBuildings: int
    YearBuilt: int
    HasGas: bool
    HasElectricity: bool
    HasSteam: bool
    HasParking: bool
    UsageCount: str
    PropertyTypeGrouped: str
    PrimaryPropertyType: str
    Neighborhood: str

# Endpoint de prédiction
@app.post("/predict")
def predict(data: InputData, db: Session = Depends(get_db)):
    # Convertir en DataFrame
    df = pd.DataFrame([data.dict()])

    # Colonnes dérivées
    df["Age"] = 2015 - df["YearBuilt"]
    df["IsRecent"] = df["YearBuilt"] >= 2010
    df["IsLarge"] = df["PropertyGFATotal"] > 400000

    # Prédiction
    prediction = model.predict(df)[0]

    # Sauvegarder l’input dans la BDD
    db_input = Input(**data.dict())
    db.add(db_input)
    db.commit()
    db.refresh(db_input)

    # Sauvegarder la prédiction
    db_output = Output(
        input_id=db_input.id,
        prediction=float(prediction),
        timestamp=datetime.utcnow()
    )
    db.add(db_output)
    db.commit()

    return {"prediction": round(float(prediction), 2)}

# Endpoint de test
@app.get("/health")
def health_check():
    return {"status": "✅ API opérationnelle"}


