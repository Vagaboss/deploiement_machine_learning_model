# tests/test_preprocessing.py

import pandas as pd
from preprocessing import preprocess_data

def test_preprocess_data_output_shapes():
    # Création d’un mini DataFrame de test (simulé)
    df_test = pd.DataFrame({
        'BuildingType': ['NonResidential'] * 5,
        'Comments': [None]*5,
        'YearsENERGYSTARCertified': [None]*5,
        'SiteEnergyUse(kBtu)': [10000, 20000, 30000, 40000, 50000],
        'OSEBuildingID': [1, 2, 3, 4, 5],
        'PropertyGFATotal': [1000, 2000, 3000, 4000, 5000],
        'Outlier': [None]*5,
        'NumberofBuildings': [1, 1, 1, 1, 1],
        'NumberofFloors': [1, 2, 3, 4, 5],
        'ComplianceStatus': ['Compliant']*5,
        'ListOfAllPropertyUseTypes': ['Office']*5,
        'PrimaryPropertyType': ['Office']*5,
        'Neighborhood': ['A']*5,
        'YearBuilt': [2000, 1990, 1980, 2010, 2015],
        'Electricity(kWh)': [100, 200, 300, 400, 500],
        'NaturalGas(kBtu)': [1000, 2000, 3000, 4000, 5000],
        'SteamUse(kBtu)': [0, 0, 0, 0, 0],
        'PropertyGFAParking': [0, 0, 0, 0, 0],
        'DataYear': [2015]*5,
        'City': ['Seattle']*5,
        'State': ['WA']*5,
        'DefaultData': [False]*5,
        'ZipCode': [98101]*5,
        'CouncilDistrictCode': [1]*5,
        'Latitude': [47.6]*5,
        'Longitude': [-122.3]*5,
        'PropertyGFABuilding(s)': [0]*5,
        'LargestPropertyUseType': ['Office']*5,
        'SecondLargestPropertyUseType': [None]*5,
        'SecondLargestPropertyUseTypeGFA': [None]*5,
        'ThirdLargestPropertyUseType': [None]*5,
        'ThirdLargestPropertyUseTypeGFA': [None]*5,
        'ENERGYSTARScore': [50]*5,
        'NaturalGas(therms)': [100]*5,
        'GHGEmissionsIntensity': [10]*5,
        'SiteEUI(kBtu/sf)': [10]*5,
        'SiteEUIWN(kBtu/sf)': [10]*5,
        'SourceEUI(kBtu/sf)': [10]*5,
        'LargestPropertyUseTypeGFA': [100]*5,
        'Electricity(kBtu)': [1000]*5,
        'SiteEnergyUseWN(kBtu)': [9000]*5,
        'TotalGHGEmissions': [100]*5,
        'PropertyName': ['B1', 'B2', 'B3', 'B4', 'B5'],
        'Address': ['Addr1']*5,
        'TaxParcelIdentificationNumber': ['TP1']*5,
        'SourceEUIWN(kBtu/sf)': [10]*5
    })

    X_train, X_test, y_train, y_test = preprocess_data(df_test)

    # Vérifie qu’on a bien les bons formats en sortie
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert X_train.shape[0] == y_train.shape[0]
