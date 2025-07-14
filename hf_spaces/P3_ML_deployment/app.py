import gradio as gr
import pandas as pd
import joblib

# Charger le modèle
model = joblib.load("models/best_rf_pipeline.joblib")

# Fonction de prédiction
def predict_energy_use(
    primary_type, neighborhood, year_built, num_buildings, num_floors, total_area,
    usage_count, property_type_grouped,
    has_electricity, has_gas, has_steam, has_parking
):
    age = 2015 - year_built
    is_large = total_area > 400000
    is_recent = year_built >= 2010

    input_dict = {
        "PrimaryPropertyType": [primary_type],
        "Neighborhood": [neighborhood],
        "YearBuilt": [year_built],
        "NumberofBuildings": [num_buildings],
        "NumberofFloors": [num_floors],
        "PropertyGFATotal": [total_area],
        "UsageCount": [usage_count],
        "PropertyTypeGrouped": [property_type_grouped],
        "HasElectricity": [has_electricity],
        "HasGas": [has_gas],
        "HasSteam": [has_steam],
        "HasParking": [has_parking],
        "IsLarge": [is_large],
        "IsRecent": [is_recent],
        "Age": [age]
    }

    input_df = pd.DataFrame(input_dict)
    prediction = model.predict(input_df)
    return round(prediction[0], 2)

# Interface Gradio
iface = gr.Interface(
    fn=predict_energy_use,
    inputs=[
        gr.Textbox(label="Primary Property Type (ex: Office)"),
        gr.Textbox(label="Neighborhood (ex: Belltown)"),
        gr.Number(label="Year Built", precision=0),
        gr.Number(label="Number of Buildings", precision=0),
        gr.Number(label="Number of Floors", precision=0),
        gr.Number(label="Total Surface (PropertyGFATotal)"),
        gr.Radio(["0", "1"], label="Usage Count (0 = mono-usage, 1 = multi-usage)"),
        gr.Textbox(label="Property Type Grouped (ex: Office, Autre...)"),
        gr.Checkbox(label="Has Electricity"),
        gr.Checkbox(label="Has Gas"),
        gr.Checkbox(label="Has Steam"),
        gr.Checkbox(label="Has Parking")
    ],
    outputs="number",
    title="Estimation de la consommation énergétique (Seattle)"
)

iface.launch()

