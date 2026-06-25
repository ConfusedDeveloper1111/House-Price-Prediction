import pandas as pd
import numpy as np

def remove_outliers_iqr(df, columns, factor=1.5):
    df_clean = df.copy()
    removed = 0
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        before = len(df_clean)
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        removed += before - len(df_clean)
    print(f"Rows removed as outliers : {removed}")
    print(f"Rows remaining           : {len(df_clean)}")
    return df_clean

def create_new_features(df):
    df = df.copy()
    df['RoomsPerHousehold']    = df['AveRooms']    / df['AveOccup']
    df['BedroomsPerRoom']      = df['AveBedrms']   / df['AveRooms']
    df['PopulationPerHousehold'] = df['Population'] / df['AveOccup']
    print("New features created:")
    print("  - RoomsPerHousehold")
    print("  - BedroomsPerRoom")
    print("  - PopulationPerHousehold")
    return df

def cap_price(df):
    before = len(df)
    df = df[df['MedHouseVal'] < 5.0].copy()
    removed = before - len(df)
    print(f"Price-capped rows removed : {removed}")
    print(f"Rows remaining            : {len(df)}")
    return df

def run_feature_engineering(df):
    print("--- Step 1: Remove price cap ---")
    df = cap_price(df)

    print("\n--- Step 2: Remove outliers ---")
    outlier_cols = ['AveRooms', 'AveBedrms', 'Population', 'AveOccup']
    df = remove_outliers_iqr(df, outlier_cols)

    print("\n--- Step 3: Create new features ---")
    df = create_new_features(df)

    print(f"\nFinal dataset shape: {df.shape}")
    print("Feature engineering complete ✅")
    return df
