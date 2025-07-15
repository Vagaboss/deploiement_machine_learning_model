# tests/test_preprocessing.py
import pytest
import pandas as pd
from src.preprocessing import preprocess_data

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
        'BuildingType': ['NonResidential'] * 8,
        'Comments': [None] * 8,
        'YearsENERGYSTARCertified': [None] * 8,
        'SiteEnergyUse(kBtu)': [-1000, 0, 1000, 1500, 2000, 5000, 6000, 7000],  # les 3 dernières sont valides
        'OSEBuildingID': [1, 2, 3, 4, 5, 6, 7, 8],
        'PropertyGFATotal': [1000] * 8,
        'Outlier': [None] * 8,
        'NumberofBuildings': [1] * 8,
        'NumberofFloors': [1] * 8,
        'ComplianceStatus': ['Compliant'] * 8,
        'ListOfAllPropertyUseTypes': ['Office'] * 8,
        'PrimaryPropertyType': ['Office'] * 8,
        'Neighborhood': ['A'] * 8,
        'YearBuilt': [2000] * 8,
        'Electricity(kWh)': [100] * 8,
        'NaturalGas(kBtu)': [100] * 8,
        'SteamUse(kBtu)': [0] * 8,
        'PropertyGFAParking': [0] * 8,
        'DataYear': [2015] * 8,
        'City': ['Seattle'] * 8,
        'State': ['WA'] * 8,
        'DefaultData': [False] * 8,
        'ZipCode': [98101] * 8,
        'CouncilDistrictCode': [1] * 8,
        'Latitude': [47.6] * 8,
        'Longitude': [-122.3] * 8,
        'PropertyGFABuilding(s)': [1000] * 8,
        'LargestPropertyUseType': ['Office'] * 8,
        'SecondLargestPropertyUseType': [None] * 8,
        'SecondLargestPropertyUseTypeGFA': [None] * 8,
        'ThirdLargestPropertyUseType': [None] * 8,
        'ThirdLargestPropertyUseTypeGFA': [None] * 8,
        'ENERGYSTARScore': [50] * 8,
        'NaturalGas(therms)': [100] * 8,
        'GHGEmissionsIntensity': [10] * 8,
        'SiteEUI(kBtu/sf)': [10] * 8,
        'SiteEUIWN(kBtu/sf)': [10] * 8,
        'SourceEUI(kBtu/sf)': [10] * 8,
        'LargestPropertyUseTypeGFA': [100] * 8,
        'Electricity(kBtu)': [1000] * 8,
        'SiteEnergyUseWN(kBtu)': [9000] * 8,
        'TotalGHGEmissions': [100] * 8,
        'PropertyName': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
        'Address': ['Addr'] * 8,
        'TaxParcelIdentificationNumber': ['TP'] * 8,
        'SourceEUIWN(kBtu/sf)': [10] * 8
    })

    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Vérifie que seules les lignes avec SiteEnergyUse(kBtu) > 0 ont été gardées
    y_combined = pd.concat([y_train, y_test])
    assert all(y_combined > 0), "Certaines valeurs d'énergie <= 0 sont toujours présentes"
    assert len(y_combined) >= 2, "Pas assez de lignes conservées pour split"





def test_very_small_dataset_filtered_out():
    # Donnée volontairement non conforme : SiteEnergyUse(kBtu) <= 0
    df = pd.DataFrame({
        'BuildingType': ['NonResidential'],
        'SiteEnergyUse(kBtu)': [0],  # Cette valeur sera filtrée
        'OSEBuildingID': [1],
        'PropertyGFATotal': [1000],
        'Outlier': [None],
        'NumberofBuildings': [1],
        'NumberofFloors': [1],
        'ComplianceStatus': ['Compliant'],
        'ListOfAllPropertyUseTypes': ['Office'],
        'PrimaryPropertyType': ['Office'],
        'Neighborhood': ['A'],
        'YearBuilt': [2000],
        'Electricity(kWh)': [100],
        'NaturalGas(kBtu)': [1000],
        'SteamUse(kBtu)': [0],
        'PropertyGFAParking': [0],
        'DataYear': [2015],
        'City': ['Seattle'],
        'State': ['WA'],
        'DefaultData': [False],
        'ZipCode': [98101],
        'CouncilDistrictCode': [1],
        'Latitude': [47.6],
        'Longitude': [-122.3],
        'PropertyGFABuilding(s)': [1000],
        'LargestPropertyUseType': ['Office'],
        'SecondLargestPropertyUseType': [None],
        'SecondLargestPropertyUseTypeGFA': [None],
        'ThirdLargestPropertyUseType': [None],
        'ThirdLargestPropertyUseTypeGFA': [None],
        'ENERGYSTARScore': [50],
        'NaturalGas(therms)': [100],
        'GHGEmissionsIntensity': [10],
        'SiteEUI(kBtu/sf)': [10],
        'SiteEUIWN(kBtu/sf)': [10],
        'SourceEUI(kBtu/sf)': [10],
        'LargestPropertyUseTypeGFA': [100],
        'Electricity(kBtu)': [1000],
        'SiteEnergyUseWN(kBtu)': [9000],
        'TotalGHGEmissions': [100],
        'PropertyName': ['B1'],
        'Address': ['Addr'],
        'TaxParcelIdentificationNumber': ['TP'],
        'SourceEUIWN(kBtu/sf)': [10],
        'Comments': [None],
        'YearsENERGYSTARCertified': [None]
    })

    # On attend une erreur car train_test_split n’aura aucune ligne à splitter
    with pytest.raises(ValueError, match="train set will be empty"):
        preprocess_data(df)

