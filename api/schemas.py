from pydantic import BaseModel, Field
from typing import Optional

# === Schéma pour l'entrée utilisateur ===
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

# === Schéma pour l'output ===
class OutputData(BaseModel):
    prediction: float = Field(..., description="Prédiction de la consommation d'énergie (kBtu)")

# === Optionnel : Schéma pour la base (lecture possible depuis SQL) ===
class InputFromDB(InputData):
    id: int

    class Config:
        orm_mode = True
