
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.preprocessing import preprocess_data

# Charger les données brutes
df = pd.read_csv("data/2016_Building_Energy_Benchmarking.csv")

# Appliquer le prétraitement
X_train, X_test, y_train, y_test = preprocess_data(df)

# Charger le modèle sauvegardé
model = joblib.load("models/best_random_forest_model.joblib")

# Prédictions
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

# Affichage des métriques
print("\n=== ÉVALUATION DU MODÈLE ===")

# Test
print("TEST R²:", r2_score(y_test, y_pred_test))
print("TEST MAE:", mean_absolute_error(y_test, y_pred_test))
print("TEST RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))

# Train
print("\nTRAIN R²:", r2_score(y_train, y_pred_train))
print("TRAIN MAE:", mean_absolute_error(y_train, y_pred_train))
print("TRAIN RMSE:", np.sqrt(mean_squared_error(y_train, y_pred_train)))
