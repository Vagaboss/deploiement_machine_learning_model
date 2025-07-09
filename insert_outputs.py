import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session
from api.models import Input, Output
from api.db import SessionLocal
import joblib
import os

# Charger le modèle
model = joblib.load("models/best_rf_pipeline.joblib")

# Charger la base
session = SessionLocal()

# Récupérer toutes les entrées Input
inputs = session.query(Input).all()

# Pour chaque ligne Input, faire une prédiction et enregistrer dans Output
for row in inputs:
    # Transformer les données en DataFrame
    input_data = {
        "PropertyGFATotal": row.PropertyGFATotal,
        "NumberofFloors": row.NumberofFloors,
        "NumberofBuildings": row.NumberofBuildings,
        "YearBuilt": row.YearBuilt,
        "HasGas": row.HasGas,
        "HasElectricity": row.HasElectricity,
        "HasSteam": row.HasSteam,
        "HasParking": row.HasParking,
        "UsageCount": row.UsageCount,
        "PropertyTypeGrouped": row.PropertyTypeGrouped,
        "PrimaryPropertyType": row.PrimaryPropertyType,
        "Neighborhood": row.Neighborhood,
    }

    df = pd.DataFrame([input_data])

    # Ajouter les colonnes calculées
    df["Age"] = 2015 - df["YearBuilt"]
    df["IsRecent"] = df["YearBuilt"] >= 2010
    df["IsLarge"] = df["PropertyGFATotal"] > 400000

    # Faire la prédiction
    prediction = model.predict(df)[0]

    # Créer une nouvelle ligne Output
    output = Output(
        input_id=row.id,
        prediction=float(prediction),
        timestamp=datetime.utcnow()
    )

    # Ajouter à la session
    session.add(output)

# Commit pour insérer en base
session.commit()
session.close()
print("✅ Prédictions insérées dans la table outputs.")
