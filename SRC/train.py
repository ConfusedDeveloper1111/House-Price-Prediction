import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def prepare_data(df):
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print(f"Train size : {X_train.shape[0]} rows")
    print(f"Test size  : {X_test.shape[0]} rows")
    print(f"Features   : {X_train.shape[1]} columns")
    print("Data prepared ✅")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()

def train_linear_regression(X_train, y_train):
    print("\nTraining Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='r2')
    print(f"Cross-Val R² scores : {scores.round(4)}")
    print(f"Mean CV R²          : {scores.mean():.4f}")
    print("Linear Regression trained ✅")
    return lr

def train_random_forest(X_train, y_train):
    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
    print(f"Cross-Val R² scores : {scores.round(4)}")
    print(f"Mean CV R²          : {scores.mean():.4f}")
    print("Random Forest trained ✅")
    return rf

def train_xgboost(X_train, y_train):
    print("\nTraining XGBoost...")
    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    scores = cross_val_score(xgb, X_train, y_train, cv=5, scoring='r2')
    print(f"Cross-Val R² scores : {scores.round(4)}")
    print(f"Mean CV R²          : {scores.mean():.4f}")
    print("XGBoost trained ✅")
    return xgb

def save_models(models_dict, scaler, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    for name, model in models_dict.items():
        path = f"{models_dir}/{name}.joblib"
        joblib.dump(model, path)
        print(f"Saved {name} → {path}")
    joblib.dump(scaler, f"{models_dir}/scaler.joblib")
    print(f"Saved scaler → {models_dir}/scaler.joblib")
    print("All models saved ✅")
