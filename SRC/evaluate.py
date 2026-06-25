import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test, name):
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    print(f"\n--- {name} ---")
    print(f"MAE  : {mae:.4f}  (~${mae*100000:,.0f} average error)")
    print(f"RMSE : {rmse:.4f}  (~${rmse*100000:,.0f} root mean sq error)")
    print(f"R²   : {r2:.4f}")
    return preds, mae, rmse, r2

def plot_actual_vs_predicted(y_test, preds, name, save_path):
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, preds, alpha=0.3, color='teal', s=5)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Fit')
    plt.xlabel("Actual Price ($100,000s)")
    plt.ylabel("Predicted Price ($100,000s)")
    plt.title(f"{name}: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved → {save_path} ✅")

def plot_residuals(y_test, preds, name, save_path):
    residuals = y_test - preds
    plt.figure(figsize=(8, 5))
    plt.scatter(preds, residuals, alpha=0.3, color='steelblue', s=5)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel("Predicted Price ($100,000s)")
    plt.ylabel("Residuals")
    plt.title(f"{name}: Residual Plot")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved → {save_path} ✅")

def plot_feature_importance(model, feature_names, save_path):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances.sort_values().plot(kind='barh', figsize=(9, 6), color='steelblue')
    plt.title("Feature Importance (XGBoost)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved → {save_path} ✅")

def plot_model_comparison(results_dict, save_path):
    names  = list(results_dict.keys())
    maes   = [results_dict[n]['mae']  for n in names]
    rmses  = [results_dict[n]['rmse'] for n in names]
    r2s    = [results_dict[n]['r2']   for n in names]

    x = range(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].bar(names, maes,  color='steelblue')
    axes[0].set_title("MAE (lower is better)")
    axes[0].set_ylabel("MAE")

    axes[1].bar(names, rmses, color='teal')
    axes[1].set_title("RMSE (lower is better)")
    axes[1].set_ylabel("RMSE")

    axes[2].bar(names, r2s,   color='seagreen')
    axes[2].set_title("R² Score (higher is better)")
    axes[2].set_ylabel("R²")

    for ax in axes:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15)

    plt.suptitle("Model Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved → {save_path} ✅")
