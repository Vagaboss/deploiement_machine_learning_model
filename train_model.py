# train_model.py

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

def train_model(df, save_model_path="models/best_rf_pipeline.joblib", save_csv_path="data/cleaned_data.csv"):
    # 1. Nettoyage
    df = df[df["BuildingType"] == "NonResidential"]
    df = df[df["SiteEnergyUse(kBtu)"] > 0]
    df = df[df["OSEBuildingID"] != 25772]
    df["EUI"] = df["SiteEnergyUse(kBtu)"] / df["PropertyGFATotal"]
    df = df[df["EUI"] <= 500]
    df = df[df["EUI"] >= 3.7]
    df = df[(df["NumberofBuildings"] != 0) & (df["NumberofFloors"] != 0)]
    df = df[df["ComplianceStatus"] != "Error - Correct Default Data"]

    df = df.drop(columns=[
        "Comments", "YearsENERGYSTARCertified", "Outlier", "DataYear", "City", "State",
        "BuildingType", "DefaultData", "ComplianceStatus"
    ], errors="ignore")

    # 2. Feature engineering
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

    # 3. X / y
    y = df["SiteEnergyUse(kBtu)"]
    X = df[[
        'PrimaryPropertyType', 'Neighborhood', 'YearBuilt', 'NumberofBuildings',
        'NumberofFloors', 'PropertyGFATotal', 'UsageCount', 'PropertyTypeGrouped',
        'HasElectricity', 'HasGas', 'HasSteam', 'HasParking', 'IsLarge', 'IsRecent', 'Age'
    ]]

    # 4. Splitt
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Pipeline
    num_cols = ['NumberofBuildings', 'NumberofFloors', 'PropertyGFATotal', 'Age']
    bool_cols = ['HasElectricity', 'HasGas', 'HasSteam', 'HasParking', 'IsLarge', 'IsRecent', 'YearBuilt']
    cat_cols = ['PrimaryPropertyType', 'Neighborhood', 'PropertyTypeGrouped', 'UsageCount']

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("bool", "passthrough", bool_cols),
        ("cat", ce.BinaryEncoder(), cat_cols)
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42))
    ])

    # 6. GridSearch
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

     # Sauvegarde du pipeline
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    joblib.dump(best_pipeline, save_model_path)

    # Sauvegarde du dataset nettoyé (features + target)
    df_cleaned = X.copy()
    df_cleaned["SiteEnergyUse(kBtu)"] = y
    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
    df_cleaned.to_csv(save_csv_path, index=False)

    return best_pipeline, df_cleaned, grid_search.best_params_, y_test, y_pred_test, y_train, y_pred_train

# 9. Pour exécuter le script en local
if __name__ == "__main__":
    df = pd.read_csv("data/2016_Building_Energy_Benchmarking.csv")
    model, df_cleaned, params, y_test, y_pred_test, y_train, y_pred_train = train_model(df)

    print("\n=== RANDOM FOREST REGRESSOR (Pipeline complet) ===")
    print("Meilleurs hyperparamètres :", params)
    print("✅ Modèle et dataset nettoyé sauvegardé dans data/cleaned_data.csv")
    print("TEST R²:", r2_score(y_test, y_pred_test))
    print("TEST MAE:", mean_absolute_error(y_test, y_pred_test))
    print("TEST RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
    print("TRAIN R²:", r2_score(y_train, y_pred_train))
    print("TRAIN MAE:", mean_absolute_error(y_train, y_pred_train))
    print("TRAIN RMSE:", np.sqrt(mean_squared_error(y_train, y_pred_train)))




