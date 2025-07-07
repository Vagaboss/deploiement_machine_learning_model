import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import category_encoders as ce

# 1. Chargement des données
df = pd.read_csv("data/2016_Building_Energy_Benchmarking.csv")

# 2. Nettoyage des données
df = df[df["BuildingType"] == "NonResidential"]
df = df[df["SiteEnergyUse(kBtu)"] > 0]
df = df[df["OSEBuildingID"] != 25772]
df["EUI"] = df["SiteEnergyUse(kBtu)"] / df["PropertyGFATotal"]
df = df[df["EUI"] <= 500]
df = df[df["EUI"] >= 3.7]
df = df[(df["NumberofBuildings"] != 0) & (df["NumberofFloors"] != 0)]
df = df[df["ComplianceStatus"] != "Error - Correct Default Data"]

# Suppression des colonnes inutiles
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

# UsageCount
df["UsageCount"] = df["ListOfAllPropertyUseTypes"].apply(
    lambda x: "1" if isinstance(x, str) and "," in x else "0"
)

# PropertyTypeGrouped
rare_types = df["PrimaryPropertyType"].value_counts()[df["PrimaryPropertyType"].value_counts() < 50].index
df["PropertyTypeGrouped"] = df["PrimaryPropertyType"].replace(rare_types, "Autre")

# 4. Cible + variables explicatives
y = df["SiteEnergyUse(kBtu)"]
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

# 5. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Définition des groupes de colonnes
num_cols = [
    'NumberofBuildings', 'NumberofFloors', 'PropertyGFATotal', 'Age'
]

bool_cols = [
    'HasElectricity', 'HasGas', 'HasSteam', 'HasParking', 'IsLarge', 'IsRecent', 'YearBuilt'
]

cat_cols = [
    'PrimaryPropertyType', 'Neighborhood', 'PropertyTypeGrouped', 'UsageCount'
]

# 7. Pipeline de prétraitement
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("bool", "passthrough", bool_cols),
    ("cat", ce.BinaryEncoder(), cat_cols)
])

# 8. Pipeline complet
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# 9. Recherche d'hyperparamètres
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [10, 20],
    'regressor__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)
best_pipeline = grid_search.best_estimator_

# 10. Évaluation
y_pred_train = best_pipeline.predict(X_train)
y_pred_test = best_pipeline.predict(X_test)

print("\n=== RANDOM FOREST REGRESSOR (Pipeline complet) ===")
print("Meilleurs hyperparamètres :", grid_search.best_params_)
print("TEST R²:", r2_score(y_test, y_pred_test))
print("TEST MAE:", mean_absolute_error(y_test, y_pred_test))
print("TEST RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("TRAIN R²:", r2_score(y_train, y_pred_train))
print("TRAIN MAE:", mean_absolute_error(y_train, y_pred_train))
print("TRAIN RMSE:", np.sqrt(mean_squared_error(y_train, y_pred_train)))

# 11. Sauvegarde du pipeline
os.makedirs("models", exist_ok=True)
joblib.dump(best_pipeline, "models/best_rf_pipeline.joblib")
