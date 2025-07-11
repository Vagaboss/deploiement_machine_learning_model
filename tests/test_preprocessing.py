# tests/test_preprocessing.py
import pytest
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



def test_missing_column_building_type():
    df = pd.DataFrame({
        'SiteEnergyUse(kBtu)': [10000],
        'PropertyGFATotal': [2000],
        # 'BuildingType' est volontairement absent
    })

    with pytest.raises(KeyError):
        preprocess_data(df)


def test_filter_negative_energy():
    df = pd.DataFrame({
        'BuildingType': ['NonResidential']*4,
        'Comments': [None]*4,
        'YearsENERGYSTARCertified': [None]*4,
        'SiteEnergyUse(kBtu)': [-1000, 0, 2000, 5000],  # dernière ligne valide
        'OSEBuildingID': [1, 2, 3, 4],
        'PropertyGFATotal': [1000, 2000, 3000, 4000],
        'Outlier': [None]*4,
        'NumberofBuildings': [1]*4,
        'NumberofFloors': [1]*4,
        'ComplianceStatus': ['Compliant']*4,
        'ListOfAllPropertyUseTypes': ['Office']*4,
        'PrimaryPropertyType': ['Office']*4,
        'Neighborhood': ['A']*4,
        'YearBuilt': [2000]*4,
        'Electricity(kWh)': [100]*4,
        'NaturalGas(kBtu)': [100]*4,
        'SteamUse(kBtu)': [0]*4,
        'PropertyGFAParking': [0]*4,
        'DataYear': [2015]*4,
        'City': ['Seattle']*4,
        'State': ['WA']*4,
        'DefaultData': [False]*4,
        'ZipCode': [98101]*4,
        'CouncilDistrictCode': [1]*4,
        'Latitude': [47.6]*4,
        'Longitude': [-122.3]*4,
        'PropertyGFABuilding(s)': [0]*4,
        'LargestPropertyUseType': ['Office']*4,
        'SecondLargestPropertyUseType': [None]*4,
        'SecondLargestPropertyUseTypeGFA': [None]*4,
        'ThirdLargestPropertyUseType': [None]*4,
        'ThirdLargestPropertyUseTypeGFA': [None]*4,
        'ENERGYSTARScore': [50]*4,
        'NaturalGas(therms)': [100]*4,
        'GHGEmissionsIntensity': [10]*4,
        'SiteEUI(kBtu/sf)': [10]*4,
        'SiteEUIWN(kBtu/sf)': [10]*4,
        'SourceEUI(kBtu/sf)': [10]*4,
        'LargestPropertyUseTypeGFA': [100]*4,
        'Electricity(kBtu)': [1000]*4,
        'SiteEnergyUseWN(kBtu)': [9000]*4,
        'TotalGHGEmissions': [100]*4,
        'PropertyName': ['B1', 'B2', 'B3', 'B4'],
        'Address': ['Addr']*4,
        'TaxParcelIdentificationNumber': ['TP']*4,
        'SourceEUIWN(kBtu/sf)': [10]*4
    })

    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Vérifie que toutes les valeurs d'énergie restantes sont strictement > 0
    assert all(y_train > 0)
    assert all(y_test > 0)

