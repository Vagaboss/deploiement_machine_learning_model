import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# 1. Charger les données
df = pd.read_csv("data/2016_Building_Energy_Benchmarking.csv")

# 2. Nettoyage identique à train_model.py
df = df[df["BuildingType"] == "NonResidential"]
df = df[df["SiteEnergyUse(kBtu)"] > 0]
df = df[df["OSEBuildingID"] != 25772]
df["EUI"] = df["SiteEnergyUse(kBtu)"] / df["PropertyGFATotal"]
df = df[df["EUI"] <= 500]
df = df[df["EUI"] >= 3.7]
df = df[(df["NumberofBuildings"] != 0) & (df["NumberofFloors"] != 0)]
df = df[df["ComplianceStatus"] != "Error - Correct Default Data"]

# Suppression de colonnes inutiles
df = df.drop(columns=[
    "Comments", "YearsENERGYSTARCertified", "Outlier", "DataYear", "City", "State",
    "BuildingType", "DefaultData", "ComplianceStatus"
], errors="ignore")

# 3. Feature engineering
df["Age"] = 2015 - df["YearBuilt"]
df["HasGas"] = df["NaturalGas(kBtu)"] > 0
df["HasElectricity"] = df["Electricity(kWh)"] > 0
df["HasSteam"] = df["SteamUse(kBtu)"] > 0
df["HasParking"] = df["PropertyGFAParking"] > 0
df["IsLarge"] = df["PropertyGFATotal"] > 400000
df["IsRecent"] = df["YearBuilt"] >= 2010
df["UsageCount"] = df["ListOfAllPropertyUseTypes"].apply(
    lambda x: "1" if isinstance(x, str) and "," in x else "0"
)
rare_types = df["PrimaryPropertyType"].value_counts()[df["PrimaryPropertyType"].value_counts() < 50].index
df["PropertyTypeGrouped"] = df["PrimaryPropertyType"].replace(rare_types, "Autre")

# 4. Variables explicatives et cible
X = df[[
    'PrimaryPropertyType',
    'Neighborhood',
    'YearBuilt',
    'NumberofBuildings',
    'NumberofFloors',
    'PropertyGFATotal',
    'UsageCount',
    'PropertyTypeGrouped',
    'HasElectricity',
    'HasGas',
    'HasSteam',
    'HasParking',
    'IsLarge',
    'IsRecent',
    'Age'
]]
y = df["SiteEnergyUse(kBtu)"]

# 5. Charger le modèle complet
model = joblib.load("models/best_rf_pipeline.joblib")

# 6. Prédiction
y_pred = model.predict(X)

# 7. Évaluation
print("\n=== ÉVALUATION DU MODÈLE (pipeline complet) ===")
print("TEST R²:", r2_score(y, y_pred))
print("TEST MAE:", mean_absolute_error(y, y_pred))
print("TEST RMSE:", np.sqrt(mean_squared_error(y, y_pred)))
