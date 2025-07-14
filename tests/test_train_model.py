import pandas as pd
import os
from train_model import train_model
import pytest
def test_train_model_runs(tmp_path):
    # ✅ Données minimales de test
    df_test = pd.DataFrame({
    'BuildingType': ['NonResidential'] * 50,
    'SiteEnergyUse(kBtu)': np.random.randint(10000, 30000, 50),
    'OSEBuildingID': range(50),
    'PropertyGFATotal': [1000]*50,
    'Outlier': [None]*50,
    'NumberofBuildings': [1]*50,
    'NumberofFloors': [1]*50,
    'ComplianceStatus': ['Compliant']*50,
    'ListOfAllPropertyUseTypes': ['Office']*50,
    'PrimaryPropertyType': ['Office']*50,
    'Neighborhood': ['A']*50,
    'YearBuilt': [2000]*50,
    'Electricity(kWh)': [100]*50,
    'NaturalGas(kBtu)': [100]*50,
    'SteamUse(kBtu)': [0]*50,
    'PropertyGFAParking': [0]*50,
    'DataYear': [2015]*50,
    'City': ['Seattle']*50,
    'State': ['WA']*50,
    'DefaultData': [False]*50,
    'ZipCode': [98101]*50,
    'CouncilDistrictCode': [1]*50,
    'Latitude': [47.6]*50,
    'Longitude': [-122.3]*50,
    'PropertyGFABuilding(s)': [1000]*50,
    'LargestPropertyUseType': ['Office']*50,
    'SecondLargestPropertyUseType': [None]*50,
    'SecondLargestPropertyUseTypeGFA': [None]*50,
    'ThirdLargestPropertyUseType': [None]*50,
    'ThirdLargestPropertyUseTypeGFA': [None]*50,
    'ENERGYSTARScore': [50]*50,
    'NaturalGas(therms)': [100]*50,
    'GHGEmissionsIntensity': [10]*50,
    'SiteEUI(kBtu/sf)': [10]*50,
    'SiteEUIWN(kBtu/sf)': [10]*50,
    'SourceEUI(kBtu/sf)': [10]*50,
    'LargestPropertyUseTypeGFA': [100]*50,
    'Electricity(kBtu)': [1000]*50,
    'SiteEnergyUseWN(kBtu)': [9000]*50,
    'TotalGHGEmissions': [100]*50,
    'PropertyName': ['B']*50,
    'Address': ['Addr']*50,
    'TaxParcelIdentificationNumber': ['TP']*50,
    'SourceEUIWN(kBtu/sf)': [10]*50,
    'Comments': [None]*50,
    'YearsENERGYSTARCertified': [None]*50
})

    # Dossiers temporaires pour les fichiers
    model_path = tmp_path / "best_model.joblib"
    csv_path = tmp_path / "cleaned_data.csv"

    model, df_cleaned, params, y_test, y_pred_test, y_train, y_pred_train = train_model(
        df_test,
        save_model_path=str(model_path),
        save_csv_path=str(csv_path)
    )


    # Vérifications
    assert model is not None
    assert df_cleaned.shape[0] > 0
    assert os.path.exists(model_path)
    assert os.path.exists(csv_path)
    assert "regressor__n_estimators" in params



import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def test_model_metrics(tmp_path):
    df_test = pd.DataFrame({
    'BuildingType': ['NonResidential'] * 50,
    'SiteEnergyUse(kBtu)': np.random.randint(10000, 30000, 50),
    'OSEBuildingID': range(50),
    'PropertyGFATotal': [1000]*50,
    'Outlier': [None]*50,
    'NumberofBuildings': [1]*50,
    'NumberofFloors': [1]*50,
    'ComplianceStatus': ['Compliant']*50,
    'ListOfAllPropertyUseTypes': ['Office']*50,
    'PrimaryPropertyType': ['Office']*50,
    'Neighborhood': ['A']*50,
    'YearBuilt': [2000]*50,
    'Electricity(kWh)': [100]*50,
    'NaturalGas(kBtu)': [100]*50,
    'SteamUse(kBtu)': [0]*50,
    'PropertyGFAParking': [0]*50,
    'DataYear': [2015]*50,
    'City': ['Seattle']*50,
    'State': ['WA']*50,
    'DefaultData': [False]*50,
    'ZipCode': [98101]*50,
    'CouncilDistrictCode': [1]*50,
    'Latitude': [47.6]*50,
    'Longitude': [-122.3]*50,
    'PropertyGFABuilding(s)': [1000]*50,
    'LargestPropertyUseType': ['Office']*50,
    'SecondLargestPropertyUseType': [None]*50,
    'SecondLargestPropertyUseTypeGFA': [None]*50,
    'ThirdLargestPropertyUseType': [None]*50,
    'ThirdLargestPropertyUseTypeGFA': [None]*50,
    'ENERGYSTARScore': [50]*50,
    'NaturalGas(therms)': [100]*50,
    'GHGEmissionsIntensity': [10]*50,
    'SiteEUI(kBtu/sf)': [10]*50,
    'SiteEUIWN(kBtu/sf)': [10]*50,
    'SourceEUI(kBtu/sf)': [10]*50,
    'LargestPropertyUseTypeGFA': [100]*50,
    'Electricity(kBtu)': [1000]*50,
    'SiteEnergyUseWN(kBtu)': [9000]*50,
    'TotalGHGEmissions': [100]*50,
    'PropertyName': ['B']*50,
    'Address': ['Addr']*50,
    'TaxParcelIdentificationNumber': ['TP']*50,
    'SourceEUIWN(kBtu/sf)': [10]*50,
    'Comments': [None]*50,
    'YearsENERGYSTARCertified': [None]*50
})


    from train_model import train_model

    model_path = tmp_path / "model.joblib"
    csv_path = tmp_path / "cleaned.csv"

    model, df_cleaned, params, y_test, y_pred_test, y_train, y_pred_train = train_model(
        df_test, save_model_path=str(model_path), save_csv_path=str(csv_path)
    )

    # Vérifie que les métriques sont bien calculées
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    for metric in [r2, mae, rmse]:
        assert isinstance(metric, float)
        assert not np.isnan(metric)
        assert not np.isinf(metric)




def test_train_model_missing_column():
    # Jeu de données avec une colonne essentielle manquante
    df_missing_col = pd.DataFrame({
        'BuildingType': ['NonResidential'],
        'SiteEnergyUse(kBtu)': [10000],
        'OSEBuildingID': [1],
        # 'PropertyGFATotal' est manquant ici
        'NumberofBuildings': [1],
        'NumberofFloors': [1],
        'ComplianceStatus': ['Compliant'],
        'ListOfAllPropertyUseTypes': ['Office'],
        'PrimaryPropertyType': ['Office'],
        'Neighborhood': ['A'],
        'YearBuilt': [2000],
        'Electricity(kWh)': [100],
        'NaturalGas(kBtu)': [100],
        'SteamUse(kBtu)': [0],
        'PropertyGFAParking': [0],
    })

    # On attend une ValueError ou KeyError
    with pytest.raises(KeyError):
        train_model(df_missing_col)






def test_train_model_with_invalid_input():
    invalid_input = "ceci n'est pas un DataFrame"

    with pytest.raises(TypeError):
        train_model(invalid_input)

