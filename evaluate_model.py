
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Charger les données
df = pd.read_csv("data/2016_Building_Energy_Benchmarking.csv")

# Même nettoyage minimal qu'avant
df = df[df["BuildingType"] == "NonResidential"]
df = df[df["SiteEnergyUse(kBtu)"] > 0]
df = df[df["OSEBuildingID"] != 25772]
df["EUI"] = df["SiteEnergyUse(kBtu)"] / df["PropertyGFATotal"]
df = df[df["EUI"] <= 500]
df = df[df["EUI"] >= 3.7]
df = df[(df["NumberofBuildings"] != 0) & (df["NumberofFloors"] != 0)]
df = df[df["ComplianceStatus"] != "Error - Correct Default Data"]
df["Age"] = 2015 - df["YearBuilt"]
df["HasGas"] = df["NaturalGas(kBtu)"] > 0
df["HasElectricity"] = df["Electricity(kWh)"] > 0

# Variables explicatives et cible
X = df[["PropertyGFATotal", "NumberofFloors", "Age", "HasGas", "HasElectricity",
        "PrimaryPropertyType", "Neighborhood", "PropertyTypeGrouped"]]
y = df["SiteEnergyUse(kBtu)"]

# Charger le modèle complet
model = joblib.load("models/best_rf_pipeline.joblib")

# Prédictions
y_pred = model.predict(X)

# Évaluation
print("\n=== ÉVALUATION DU MODÈLE (pipeline complet) ===")
print("TEST R²:", r2_score(y, y_pred))
print("TEST MAE:", mean_absolute_error(y, y_pred))
print("TEST RMSE:", np.sqrt(mean_squared_error(y, y_pred)))
