
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
from src.preprocessing import preprocess_data

# Chargement des données
df = pd.read_csv("data/2016_Building_Energy_Benchmarking.csv")


# Prétraitement
X_train, X_test, y_train, y_test = preprocess_data(df)

# Définition de la grille d'hyperparamètres
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

# Entraînement avec GridSearchCV
grid_rf = GridSearchCV(RandomForestRegressor(random_state=42),
                       param_grid_rf,
                       cv=5,
                       scoring='r2')
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

# Prédictions
y_pred_rf_train = best_rf.predict(X_train)
y_pred_rf_test = best_rf.predict(X_test)

# Affichage des scores
print("\n=== RANDOM FOREST REGRESSOR ===")
print("Best params:", grid_rf.best_params_)
print("TEST R²:", r2_score(y_test, y_pred_rf_test))
print("TEST MAE:", mean_absolute_error(y_test, y_pred_rf_test))
print("TEST RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf_test)))
print("TRAIN R²:", r2_score(y_train, y_pred_rf_train))
print("TRAIN MAE:", mean_absolute_error(y_train, y_pred_rf_train))
print("TRAIN RMSE:", np.sqrt(mean_squared_error(y_train, y_pred_rf_train)))

# Sauvegarde du modèle
import os
os.makedirs("models", exist_ok=True)
joblib.dump(best_rf, "models/best_random_forest_model.joblib")
