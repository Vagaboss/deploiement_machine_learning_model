from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

# === Schéma pour l'entrée utilisateur (requête API) ===
class InputData(BaseModel):
    PropertyGFATotal: float = Field(..., description="Surface totale du bâtiment en pieds carrés")
    NumberofFloors: int = Field(..., description="Nombre d'étages")
    NumberofBuildings: int = Field(..., description="Nombre total de bâtiments")
    YearBuilt: int = Field(..., description="Année de construction")
    HasGas: bool = Field(..., description="Présence de gaz")
    HasElectricity: bool = Field(..., description="Présence d'électricité")
    HasSteam: bool = Field(..., description="Présence de vapeur")
    HasParking: bool = Field(..., description="Présence de parking")
    UsageCount: str = Field(..., description="Mono-usage = '0', Multi-usage = '1'")
    PropertyTypeGrouped: str = Field(..., description="Catégorie du type de propriété")
    PrimaryPropertyType: str = Field(..., description="Type principal du bâtiment")
    Neighborhood: str = Field(..., description="Quartier de Seattle")

# === Schéma pour la réponse de prédiction (API) ===
class OutputData(BaseModel):
    prediction: float = Field(..., description="Prédiction de la consommation d'énergie (kBtu)")

# === Schéma ORM pour lecture d'une entrée depuis la BDD ===
class InputFromDB(InputData):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True

# === Schéma ORM pour lecture d'une prédiction depuis la BDD ===
class OutputFromDB(BaseModel):
    id: int
    input_id: int
    prediction: float
    timestamp: datetime

    class Config:
        from_attributes = True

# === Schéma combiné pour l’historique (input + output) ===
class HistoryRecord(BaseModel):
    input: InputFromDB
    output: OutputFromDB


