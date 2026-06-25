import gradio as gr
import joblib
import pandas as pd
import numpy as np
import os

# Load model and scaler
model  = joblib.load("xgboost.joblib")
scaler = joblib.load("scaler.joblib")

def predict_house_price(MedInc, HouseAge, AveRooms, AveBedrms,
                        Population, AveOccup, Latitude, Longitude):
    try:
        if AveRooms <= 0 or AveOccup <= 0:
            return "❌ AveRooms and AveOccup must be greater than 0"
        if not (-124.5 <= Longitude <= -114.0):
            return "❌ Longitude out of California range (-124.5 to -114.0)"
        if not (32.5 <= Latitude <= 42.0):
            return "❌ Latitude out of California range (32.5 to 42.0)"

        RoomsPerHousehold      = AveRooms   / AveOccup
        BedroomsPerRoom        = AveBedrms  / AveRooms
        PopulationPerHousehold = Population / AveOccup

        input_df = pd.DataFrame([{
            'MedInc'                 : MedInc,
            'HouseAge'               : HouseAge,
            'AveRooms'               : AveRooms,
            'AveBedrms'              : AveBedrms,
            'Population'             : Population,
            'AveOccup'               : AveOccup,
            'Latitude'               : Latitude,
            'Longitude'              : Longitude,
            'RoomsPerHousehold'      : RoomsPerHousehold,
            'BedroomsPerRoom'        : BedroomsPerRoom,
            'PopulationPerHousehold' : PopulationPerHousehold
        }])

        input_scaled = scaler.transform(input_df)
        prediction   = model.predict(input_scaled)[0]
        price        = prediction * 100000

        return (
            f"🏠 Estimated House Price : ${price:,.0f}\n"
            f"📊 Model Value (x100k)  : {prediction:.4f}\n"
            f"🔍 Top Factor           : Median Income = {MedInc}"
        )

    except Exception as e:
        return f"❌ Error: {str(e)}"

app = gr.Interface(
    fn=predict_house_price,
    inputs=[
        gr.Slider(minimum=0.5,   maximum=15.0,  value=5.0,    step=0.1,  label="Median Income (x$10,000)"),
        gr.Slider(minimum=1.0,   maximum=52.0,  value=20.0,   step=1.0,  label="House Age (years)"),
        gr.Slider(minimum=1.0,   maximum=20.0,  value=6.0,    step=0.1,  label="Average Rooms"),
        gr.Slider(minimum=0.5,   maximum=5.0,   value=1.2,    step=0.1,  label="Average Bedrooms"),
        gr.Slider(minimum=3.0,   maximum=5000.0,value=500.0,  step=10.0, label="Population"),
        gr.Slider(minimum=1.0,   maximum=10.0,  value=3.0,    step=0.1,  label="Average Occupancy"),
        gr.Slider(minimum=32.5,  maximum=42.0,  value=34.05,  step=0.01, label="Latitude"),
        gr.Slider(minimum=-124.5,maximum=-114.0,value=-118.25,step=0.01, label="Longitude"),
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="🏠 California House Price Predictor",
    description=(
        "Predict California house prices using XGBoost (R² = 0.8374).\n"
        "Adjust the sliders and click Submit to get an estimated price.\n"
        "Model trained on California Housing Dataset."
    ),
    examples=[
        [8.0, 15.0, 7.0, 1.0, 400.0,  2.5, 37.77, -122.41],
        [3.0, 30.0, 5.0, 1.2, 800.0,  3.5, 34.05, -118.25],
        [1.5, 40.0, 4.0, 1.5, 1200.0, 4.0, 36.77, -119.41],
    ],
    theme=gr.themes.Soft()
)

app.launch()
