import pandas as pd
import os
from train_model import train_model

def test_train_model_runs(tmp_path):
    # ✅ Données minimales de test
    df_test = pd.DataFrame({
        'BuildingType': ['NonResidential'] * 10,
        'SiteEnergyUse(kBtu)': [10000, 15000, 20000, 25000, 30000, 18000, 19000, 17000, 16000, 14000],
        'OSEBuildingID': list(range(10)),
        'PropertyGFATotal': [1000]*10,
        'Outlier': [None]*10,
        'NumberofBuildings': [1]*10,
        'NumberofFloors': [1]*10,
        'ComplianceStatus': ['Compliant']*10,
        'ListOfAllPropertyUseTypes': ['Office']*10,
        'PrimaryPropertyType': ['Office']*10,
        'Neighborhood': ['A']*10,
        'YearBuilt': [2000]*10,
        'Electricity(kWh)': [100]*10,
        'NaturalGas(kBtu)': [100]*10,
        'SteamUse(kBtu)': [0]*10,
        'PropertyGFAParking': [0]*10,
        'DataYear': [2015]*10,
        'City': ['Seattle']*10,
        'State': ['WA']*10,
        'DefaultData': [False]*10,
        'ZipCode': [98101]*10,
        'CouncilDistrictCode': [1]*10,
        'Latitude': [47.6]*10,
        'Longitude': [-122.3]*10,
        'PropertyGFABuilding(s)': [1000]*10,
        'LargestPropertyUseType': ['Office']*10,
        'SecondLargestPropertyUseType': [None]*10,
        'SecondLargestPropertyUseTypeGFA': [None]*10,
        'ThirdLargestPropertyUseType': [None]*10,
        'ThirdLargestPropertyUseTypeGFA': [None]*10,
        'ENERGYSTARScore': [50]*10,
        'NaturalGas(therms)': [100]*10,
        'GHGEmissionsIntensity': [10]*10,
        'SiteEUI(kBtu/sf)': [10]*10,
        'SiteEUIWN(kBtu/sf)': [10]*10,
        'SourceEUI(kBtu/sf)': [10]*10,
        'LargestPropertyUseTypeGFA': [100]*10,
        'Electricity(kBtu)': [1000]*10,
        'SiteEnergyUseWN(kBtu)': [9000]*10,
        'TotalGHGEmissions': [100]*10,
        'PropertyName': ['B']*10,
        'Address': ['Addr']*10,
        'TaxParcelIdentificationNumber': ['TP']*10,
        'SourceEUIWN(kBtu/sf)': [10]*10,
        'Comments': [None]*10,
        'YearsENERGYSTARCertified': [None]*10
    })

    # Dossiers temporaires pour les fichiers
    model_path = tmp_path / "best_model.joblib"
    csv_path = tmp_path / "cleaned_data.csv"

    model, df_cleaned, params = train_model(df_test, save_model_path=str(model_path), save_csv_path=str(csv_path))

    # Vérifications
    assert model is not None
    assert df_cleaned.shape[0] > 0
    assert os.path.exists(model_path)
    assert os.path.exists(csv_path)
    assert "regressor__n_estimators" in params
