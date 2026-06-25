import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def get_basic_info(df):
    print("--- Shape ---")
    print(df.shape)
    print("--- Columns ---")
    print(df.columns.tolist())
    print("--- Data Types ---")
    print(df.dtypes)
    print("--- Missing Values ---")
    print(df.isnull().sum())
    print("--- Basic Statistics ---")
    print(df.describe())

def get_feature_target_split(df):
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    print(f"Features: {X.shape[1]} columns")
    print(f"Target  : MedHouseVal")
    return X, y
