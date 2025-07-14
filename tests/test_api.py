# tests/test_api_predict.py
import pytest
from fastapi.testclient import TestClient

# 🔹 On importe l'app FastAPI déjà définie
from api.main import app

client = TestClient(app)


def test_predict_endpoint_ok():
    """
    Vérifie que /predict répond 200 et renvoie une prédiction numérique.
    """
    payload = {
        "PrimaryPropertyType": "Office",
        "Neighborhood": "Downtown",
        "YearBuilt": 2005,
        "NumberofBuildings": 1,
        "NumberofFloors": 5,
        "PropertyGFATotal": 12000,
        "UsageCount": "0",
        "PropertyTypeGrouped": "Office",
        "HasElectricity": True,
        "HasGas": False,
        "HasSteam": False,
        "HasParking": True
    }

    response = client.post("/predict", json=payload)

    # --- assertions ---
    assert response.status_code == 200, "L’API devrait retourner HTTP 200"
    data = response.json()

    # La clé doit exister
    assert "prediction" in data, "La réponse doit contenir une clé 'prediction'"

    # La prédiction doit être un nombre (float ou int)
    assert isinstance(data["prediction"], (float, int)), "La prédiction doit être numérique"


def test_predict_invalid_payload():
    """
    Vérifie qu’un payload incomplet renvoie une erreur 422 (validation Pydantic).
    """
    bad_payload = {
        "PrimaryPropertyType": "Office"
        # (on omet volontairement tous les autres champs requis)
    }

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422  # Unprocessable Entity


def test_predict_endpoint_wrong_type():
    client = TestClient(app)
    input_data = {
        "PrimaryPropertyType": "Office",
        "Neighborhood": "Downtown",
        "YearBuilt": "deux mille",  # mauvais type
        "NumberofBuildings": 1,
        "NumberofFloors": 2,
        "PropertyGFATotal": 10000.0,
        "UsageCount": 1,
        "PropertyTypeGrouped": "Office",
        "HasElectricity": True,
        "HasGas": True,
        "HasSteam": False,
        "HasParking": False
    }

    response = client.post("/predict", json=input_data)
    assert response.status_code == 422


