import pandas as pd
import numpy as np
import joblib

def load_model_and_scaler(models_dir):
    model  = joblib.load(f"{models_dir}/xgboost.joblib")
    scaler = joblib.load(f"{models_dir}/scaler.joblib")
    print("Model and scaler loaded ✅")
    return model, scaler

def predict_price(model, scaler, input_dict):
    expected_cols = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
        'Population', 'AveOccup', 'Latitude', 'Longitude',
        'RoomsPerHousehold', 'BedroomsPerRoom', 'PopulationPerHousehold'
    ]

    input_df = pd.DataFrame([input_dict])

    # Auto-calculate engineered features if not provided
    if 'RoomsPerHousehold' not in input_df.columns:
        input_df['RoomsPerHousehold']      = input_df['AveRooms']    / input_df['AveOccup']
        input_df['BedroomsPerRoom']        = input_df['AveBedrms']   / input_df['AveRooms']
        input_df['PopulationPerHousehold'] = input_df['Population']  / input_df['AveOccup']

    input_df = input_df[expected_cols]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    price_dollars = prediction * 100000

    print(f"\n--- Prediction Result ---")
    print(f"Predicted Value  : {prediction:.4f} (x$100,000)")
    print(f"Estimated Price  : ${price_dollars:,.0f}")
    return prediction, price_dollars

def validate_input(input_dict):
    required = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                'Population', 'AveOccup', 'Latitude', 'Longitude']
    missing = [k for k in required if k not in input_dict]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    if input_dict['AveRooms'] <= 0:
        raise ValueError("AveRooms must be greater than 0")
    if input_dict['AveOccup'] <= 0:
        raise ValueError("AveOccup must be greater than 0")
    if not (-124.5 <= input_dict['Longitude'] <= -114.0):
        raise ValueError("Longitude out of California range")
    if not (32.5 <= input_dict['Latitude'] <= 42.0):
        raise ValueError("Latitude out of California range")
    print("Input validation passed ✅")
