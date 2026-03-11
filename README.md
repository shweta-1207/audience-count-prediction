# audience-count-prediction
Predicting daily audience count across multiple locations using machine learning.

# Built for the Cinema Audience Forecasting Challenge (IIT Madras Kaggle Competition, 2025).

## Approach
- Merged data from two booking sources: BookNow (online) and CinePOS (point-of-sale)
- Engineered lag features, rolling averages, and EWMA to capture temporal demand patterns
- Trained LightGBM, XGBoost, RandomForest, and GradientBoosting with TimeSeriesSplit cross-validation
- Combined models into a weighted ensemble (LightGBM 55%, XGBoost 30%, RF 10%, GB 5%)
- Used an iterative prediction loop to generate forecasts for future dates

## Result
R² = 0.57 on cross-validation

## Files
- `Audience_Count_Prediction.ipynb` — full pipeline: EDA, feature engineering, modelling, submission
- `requirements.txt` — dependencies
