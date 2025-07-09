import pandas as pd
from sqlalchemy.orm import Session
from datetime import datetime
from api.db import engine
from api.models import Input
from sqlalchemy import insert

# 1. Charger le fichier CSV nettoyé
df = pd.read_csv("data/cleaned_data.csv")

# 2. Garder uniquement les colonnes nécessaires pour la table Input
columns = [
    'PropertyGFATotal', 'NumberofFloors', 'NumberofBuildings',
    'YearBuilt', 'HasGas', 'HasElectricity', 'HasSteam',
    'HasParking', 'UsageCount', 'PropertyTypeGrouped',
    'PrimaryPropertyType', 'Neighborhood'
]
df = df[columns]

# 3. Créer une session de connexion à la base de données
session = Session(bind=engine)

# 4. Insérer les données dans la table inputs
try:
    for _, row in df.iterrows():
        input_entry = Input(
            timestamp=datetime.utcnow(),
            PropertyGFATotal=row['PropertyGFATotal'],
            NumberofFloors=row['NumberofFloors'],
            NumberofBuildings=row['NumberofBuildings'],
            YearBuilt=row['YearBuilt'],
            HasGas=row['HasGas'],
            HasElectricity=row['HasElectricity'],
            HasSteam=row['HasSteam'],
            HasParking=row['HasParking'],
            UsageCount=row['UsageCount'],
            PropertyTypeGrouped=row['PropertyTypeGrouped'],
            PrimaryPropertyType=row['PrimaryPropertyType'],
            Neighborhood=row['Neighborhood'],
        )
        session.add(input_entry)
    session.commit()
    print("✅ Données insérées avec succès dans la table inputs.")
except Exception as e:
    session.rollback()
    print(f"❌ Une erreur est survenue : {e}")
finally:
    session.close()
