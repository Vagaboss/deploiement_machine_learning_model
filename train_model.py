
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

# 2. Prétraitement manuel léger (conservation des bonnes données)
df = df[df["BuildingType"] == "NonResidential"]
df = df[df["SiteEnergyUse(kBtu)"] > 0]
df = df[df["OSEBuildingID"] != 25772]
df["EUI"] = df["SiteEnergyUse(kBtu)"] / df["PropertyGFATotal"]
df = df[df["EUI"] <= 500]
df = df[df["EUI"] >= 3.7]
df = df[
    (df["NumberofBuildings"] != 0) & (df["NumberofFloors"] != 0)
]
df = df[df["ComplianceStatus"] != "Error - Correct Default Data"]
df = df.drop(columns=["Comments", "YearsENERGYSTARCertified", "Outlier", "DataYear", "City", "State", "BuildingType", "DefaultData", "ComplianceStatus"], errors="ignore")
df["Age"] = 2015 - df["YearBuilt"]
df["HasGas"] = df["NaturalGas(kBtu)"] > 0
df["HasElectricity"] = df["Electricity(kWh)"] > 0

# Création de la cible
y = df["SiteEnergyUse(kBtu)"]
X = df[["PropertyGFATotal", "NumberofFloors", "Age", "HasGas", "HasElectricity",
        "PrimaryPropertyType", "Neighborhood", "PropertyTypeGrouped"]]

# 3. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Définition des colonnes
num_cols = ["PropertyGFATotal", "NumberofFloors", "Age"]
bool_cols = ["HasGas", "HasElectricity"]
cat_cols = ["PrimaryPropertyType", "Neighborhood", "PropertyTypeGrouped"]

# 5. Création du préprocesseur
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("bool", "passthrough", bool_cols),
    ("cat", ce.BinaryEncoder(), cat_cols)
])

# 6. Définition du pipeline complet
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# 7. Grille de recherche
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [10, 20],
    'regressor__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)
best_pipeline = grid_search.best_estimator_

# 8. Évaluation
y_pred_train = best_pipeline.predict(X_train)
y_pred_test = best_pipeline.predict(X_test)

print("\n=== RANDOM FOREST REGRESSOR (Pipeline) ===")
print("Best params:", grid_search.best_params_)
print("TEST R²:", r2_score(y_test, y_pred_test))
print("TEST MAE:", mean_absolute_error(y_test, y_pred_test))
print("TEST RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("TRAIN R²:", r2_score(y_train, y_pred_train))
print("TRAIN MAE:", mean_absolute_error(y_train, y_pred_train))
print("TRAIN RMSE:", np.sqrt(mean_squared_error(y_train, y_pred_train)))

# 9. Sauvegarde du pipeline complet
os.makedirs("models", exist_ok=True)
joblib.dump(best_pipeline, "models/best_rf_pipeline.joblib")

