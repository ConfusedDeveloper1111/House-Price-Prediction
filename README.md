# House Price Prediction

An end-to-end machine learning project that predicts California house prices from user-provided features and serves live predictions through a Gradio app deployed on Hugging Face Spaces.

## Live Demo

**Hugging Face Space:** https://huggingface.co/spaces/JBond07/house-price-predictor

---

## Project at a Glance

<img src="./house_price_readme_assets/project-overview-pipeline.png" alt="Project overview pipeline" width="100%" />

This project follows a full ML workflow:

1. Load the California Housing dataset
2. Clean the data and engineer better features
3. Compare multiple models with cross validation
4. Select the best model
5. Deploy a live inference interface for real-time predictions

---

## What the App Does

The app takes 8 raw housing inputs:

- Median income
- House age
- Average rooms
- Average bedrooms
- Population
- Average occupancy
- Latitude
- Longitude

It then automatically creates 3 engineered features and returns a predicted house value in dollars.

---

## Dataset

- **Source:** California Housing dataset from scikit-learn
- **Rows:** 20,640
- **Columns:** 9
- **Target:** `MedHouseVal`
- **Target cap:** values above \$500k are clipped in the original dataset
- **Missing values:** none

### Main observations

- `MedInc` is the strongest predictor of house value
- Coastal locations are much more expensive than inland areas
- `AveOccup` and `Population` contain extreme outliers
- `AveRooms` and `AveBedrms` show multicollinearity

---

## Feature Engineering

<img src="./house_price_readme_assets/feature-engineering-deep-dive.png" alt="Feature engineering deep dive" width="100%" />

Three engineered features were added:

- `RoomsPerHousehold = AveRooms / AveOccup`
- `BedroomsPerRoom = AveBedrms / AveRooms`
- `PopulationPerHousehold = Population / AveOccup`

Cleaning steps included:

- IQR-based outlier removal
- Removal of capped target rows at \$500k
- Conversion from raw data to a cleaner training dataset

**Result:** 20,640 rows → 16,540 rows

---

## Geographic Insight

<img src="./house_price_readme_assets/geographic-price-heatmap.png" alt="Geographic price heatmap" width="100%" />

This visualization makes the main business insight obvious:

- Coastal California is more expensive
- Inland regions are cheaper
- Location and income are the strongest price drivers

---

## Model Comparison

<img src="./house_price_readme_assets/model-comparison-battle.png" alt="Model comparison battle" width="100%" />

Three models were trained and compared using 5-fold cross validation.

| Model | CV R² | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Linear Regression | 0.6567 | \$41,487 | \$56,132 | 0.6697 |
| Random Forest | 0.7720 | \$30,226 | \$44,326 | 0.7940 |
| **XGBoost** | **0.8256** | **\$26,801** | **\$39,385** | **0.8374** |

### Final model

- **Model:** XGBoost Regressor
- **Why it won:** best accuracy and lowest error
- **Known limitation:** performance is weaker for higher-end houses because the dataset is capped at \$500k

---

## Live Inference Interface

<img src="./house_price_readme_assets/live-app-deployment.png" alt="Live app and deployment" width="100%" />

The deployed Gradio app lets users:

- Adjust input sliders
- Try predefined examples
- Get instant price predictions
- Use the model without running local code

### Deployment

- **Frontend:** Gradio
- **Hosting:** Hugging Face Spaces
- **Artifacts saved:** model and scaler joblib files

---

## Repository Structure

```text
HousePricePrediction/
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 05_Gradio_App.ipynb
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   ├── xgboost.joblib
│   ├── random_forest.joblib
│   ├── linear_regression.joblib
│   └── scaler.joblib
├── outputs/
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_distributions.png
│   ├── outlier_boxplots.png
│   ├── geographic_distribution.png
│   ├── income_vs_price.png
│   ├── engineered_features.png
│   ├── actual_vs_predicted_*.png
│   ├── residuals_*.png
│   ├── feature_importance_xgboost.png
│   ├── model_comparison.png
│   ├── eda_summary.txt
│   ├── feature_engineering_summary.txt
│   ├── metrics_report.txt
│   └── engineered_dataset.csv
├── app.py
└── requirements.txt
```

---

## How to Run Locally

```bash
git clone <your-repo-url>
cd HousePricePrediction
pip install -r requirements.txt
python app.py
```

---

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- XGBoost
- joblib
- Gradio
- Hugging Face Spaces
- Matplotlib / Seaborn

---

## Key Takeaways

- Cleaning and feature engineering improved model quality
- Location and income dominate house price prediction
- XGBoost outperformed linear regression and random forest
- The project is fully deployable and demonstrates the full ML lifecycle

---

## Limitations

- The original dataset caps prices at \$500k
- Extremely expensive houses are underrepresented
- Feature engineering and model logic were run in a slightly messy notebook flow
- README and GitHub repository organization still need final polish

---

## Future Improvements

- Separate feature engineering, training, and evaluation into dedicated notebooks/scripts
- Add stronger model interpretability with SHAP
- Improve handling of capped price rows
- Add a cleaner experiment-tracking setup

---

## Project Summary

This project is a complete end-to-end machine learning pipeline for California house price prediction: data loading, cleaning, feature engineering, model selection, evaluation, and live deployment.

