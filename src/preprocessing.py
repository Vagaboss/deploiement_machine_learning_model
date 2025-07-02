
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import category_encoders as ce

def preprocess_data(df):
    # Filtrage des bâtiments non résidentiels
    df = df[df["BuildingType"] == "NonResidential"]

    # Suppression de colonnes inutiles
    df.drop(['Comments', 'YearsENERGYSTARCertified'], axis=1, inplace=True)

    # Nettoyage de base
    df = df[df['SiteEnergyUse(kBtu)'] > 0]
    df = df[df['OSEBuildingID'] != 25772]

    # Calcul EUI
    df['EUI'] = df['SiteEnergyUse(kBtu)'] / df['PropertyGFATotal']
    q1 = df['EUI'].quantile(0.01)
    q99 = df['EUI'].quantile(0.99)

    df['EUI_outlier'] = 'normal'
    df.loc[df['EUI'] < q1, 'EUI_outlier'] = 'bas'
    df.loc[df['EUI'] > q99, 'EUI_outlier'] = 'haut'

    # Suppression des EUI extrêmes
    df = df[df['EUI'] <= 500]
    df = df[df['EUI'] >= 3.70]

    # Suppression des outliers spécifiés dans la colonne 'Outlier'
    df_outlier_str = df[df["Outlier"].apply(lambda x: isinstance(x, str))]
    df = df[~df['OSEBuildingID'].isin(df_outlier_str['OSEBuildingID'])]
    df = df.drop(columns=['Outlier'])

    # Garder uniquement les bâtiments valides
    df = df[(df['NumberofBuildings'] != 0) & (df['NumberofFloors'] != 0)]
    df = df[df['ComplianceStatus'] != 'Error - Correct Default Data']

    # Création de variables dérivées
    df["UsageType"] = df["ListOfAllPropertyUseTypes"].apply(
        lambda x: "Multi-usage" if isinstance(x, str) and "," in x else "Mono-usage"
    )
    df["UsageCount"] = df["ListOfAllPropertyUseTypes"].apply(
        lambda x: "1" if isinstance(x, str) and "," in x else "0"
    )
    df["HasElectricity"] = df["Electricity(kWh)"] > 0
    df["HasGas"] = df["NaturalGas(kBtu)"] > 0
    df["HasSteam"] = df["SteamUse(kBtu)"] > 0
    df["HasParking"] = df["PropertyGFAParking"] > 0
    df["IsLarge"] = df["PropertyGFATotal"] > 400000
    df["IsRecent"] = df["YearBuilt"] >= 2010
    df["Age"] = 2015 - df["YearBuilt"]

    # Regroupement des types rares
    rare_types = df["PrimaryPropertyType"].value_counts()[df["PrimaryPropertyType"].value_counts() < 50].index
    df["PropertyTypeGrouped"] = df["PrimaryPropertyType"].replace(rare_types, "Autre")

    # Suppression de colonnes inutiles
    df = df.drop(columns=[
        'DataYear', 'BuildingType', 'City', 'State', 'DefaultData', 'ComplianceStatus',
        "OSEBuildingID", "ZipCode", "CouncilDistrictCode", "Latitude", "Longitude",
        "PropertyGFAParking", "PropertyGFABuilding(s)", "LargestPropertyUseType",
        "SecondLargestPropertyUseType", "SecondLargestPropertyUseTypeGFA",
        "ThirdLargestPropertyUseType", "ThirdLargestPropertyUseTypeGFA",
        "ENERGYSTARScore", "NaturalGas(therms)", "GHGEmissionsIntensity", "Electricity(kWh)",
        "SiteEUI(kBtu/sf)", "SiteEUIWN(kBtu/sf)", "SourceEUI(kBtu/sf)",
        "LargestPropertyUseTypeGFA", "Electricity(kBtu)", "SiteEnergyUseWN(kBtu)",
        "TotalGHGEmissions", "PropertyName", "Address", "TaxParcelIdentificationNumber",
        "UsageType", "ListOfAllPropertyUseTypes", 'EUI_outlier', 'SourceEUIWN(kBtu/sf)',
        'SteamUse(kBtu)', 'NaturalGas(kBtu)', 'EUI'
    ])

    # Séparation X / y
    y = df['SiteEnergyUse(kBtu)']
    X = df.drop(columns=['SiteEnergyUse(kBtu)'])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Encodage successif
    encoder1 = ce.BinaryEncoder(cols=["PrimaryPropertyType"])
    X_train = encoder1.fit_transform(X_train)
    X_test = encoder1.transform(X_test)

    encoder2 = ce.BinaryEncoder(cols=["Neighborhood"])
    X_train = encoder2.fit_transform(X_train)
    X_test = encoder2.transform(X_test)

    encoder3 = ce.BinaryEncoder(cols=["PropertyTypeGrouped"])
    X_train = encoder3.fit_transform(X_train)
    X_test = encoder3.transform(X_test)

    # Normalisation
    cols_to_scale = ['NumberofBuildings', 'NumberofFloors', 'PropertyGFATotal','Age']
    cols_passthrough = [col for col in X_train.columns if col not in cols_to_scale]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[cols_to_scale]),
                                  columns=cols_to_scale, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test[cols_to_scale]),
                                 columns=cols_to_scale, index=X_test.index)

    X_train_final = pd.concat([X_train_scaled, X_train[cols_passthrough]], axis=1)
    X_test_final = pd.concat([X_test_scaled, X_test[cols_passthrough]], axis=1)

    return X_train_final, X_test_final, y_train, y_test
