import joblib
import pandas as pd
from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, joinedload
from datetime import datetime

from api.db import SessionLocal
from api.models import Input, Output
from api.schemas import InputData, OutputData, HistoryRecord, InputFromDB, OutputFromDB
from api.models import DatasetCleaned

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

# Endpoint de prédiction
@app.post("/predict", response_model=OutputData)
def predict(data: InputData, db: Session = Depends(get_db)):
    df = pd.DataFrame([data.dict()])
    df["Age"] = 2015 - df["YearBuilt"]
    df["IsRecent"] = df["YearBuilt"] >= 2010
    df["IsLarge"] = df["PropertyGFATotal"] > 400000

    prediction = model.predict(df)[0]

    db_input = Input(**data.dict())
    db.add(db_input)
    db.commit()
    db.refresh(db_input)

    db_output = Output(
        input_id=db_input.id,
        prediction=float(prediction),
        timestamp=datetime.utcnow()
    )
    db.add(db_output)
    db.commit()

    return OutputData(prediction=round(float(prediction), 2))

# Endpoint de vérification
@app.get("/health")
def health_check():
    return {"status": "✅ API opérationnelle"}

# Endpoint historique
@app.get("/history", response_model=list[HistoryRecord])
def get_history(limit: int = 10, db: Session = Depends(get_db)):
    results = (
        db.query(Output)
        .options(joinedload(Output.input))
        .order_by(Output.id.desc())
        .limit(limit)
        .all()
    )

    history = []
    for record in results:
        input_data = InputFromDB.from_orm(record.input)
        output_data = OutputFromDB.from_orm(record)
        history.append(HistoryRecord(input=input_data, output=output_data))

    return history


# Endpoint datavisu
@app.get("/dataset")
def get_dataset(limit: int = 10, db: Session = Depends(get_db)):
    rows = db.query(DatasetCleaned).limit(limit).all()
    return [row.__dict__ for row in rows]




