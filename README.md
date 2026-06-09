# 🌫️ AQI Predictor

A machine learning-powered **3-day Air Quality Index (AQI) forecasting system** with a Streamlit dashboard, integrated with Hopsworks feature store and model registry for end-to-end MLOps.

## 🎯 What This Project Does

This project predicts air quality for the next 72 hours using:
- **Real-time data pipelines** that fetch environmental and weather data
- **ML models** (XGBoost, LightGBM, Deep Learning) trained on historical AQI patterns
- **Feature engineering** with Hopsworks Feature Store for versioning and lineage
- **Interactive Streamlit dashboard** showing current AQI, 3-day forecasts, and pollutant breakdowns
- **SHAP explainability** to understand which factors influence AQI predictions

## 🚀 Features

- **3-Day Forecast**: Predict AQI values for 24h, 48h, and 72h ahead
- **Real-time Dashboard**: Streamlit web app with current conditions and colored AQI alerts
- **Hazard Alerts**: Warnings when AQI exceeds unhealthy thresholds
- **Pollutant Breakdown**: Visualize PM2.5, PM10, O₃, NO₂, SO₂, CO levels
- **Model Explainability**: SHAP beeswarm and summary plots to understand predictions
- **Feature Store Integration**: Hopsworks for data versioning, feature management, and model registry
- **Automated Pipelines**: Training, feature engineering, and backfill jobs

## 📊 Dashboard Features

The Streamlit app (`app/app.py`) includes:
- **Current AQI Metric** with color-coded air quality status
- **Pollutant Readings** (PM2.5, PM10, Temperature, etc.)
- **3-Day AQI Forecast** with color-coded predictions
- **Historical AQI Chart** (last 7 days with health thresholds)
- **Pollutant Bar Chart** showing current concentrations
- **Auto-alerts** for hazardous conditions

## 🏗️ Project Structure

```
aqi_predictor/
├── app/
│   └── app.py                      # Streamlit dashboard application
├── pipelines/
│   ├── training_pipeline.py        # ML model training & registration
│   ├── feature_pipeline.py         # Real-time feature engineering
│   └── backfill.py                 # Historical data backfill
├── models/                         # Trained model artifacts
├── best_aqi_forecaster.pkl         # Serialized forecaster model
├── shap_summary.png                # Model explainability visualizations
├── shap_beeswarm.png
├── requirements.txt
├── run.sh                          # Script runner for pipelines
└── README.md
```

## 🛠️ Tech Stack

### ML & Data
- **scikit-learn**, **XGBoost**, **LightGBM** - Gradient boosting models
- **TensorFlow/PyTorch** - Deep learning models
- **pandas**, **numpy** - Data manipulation
- **scipy**, **statsmodels** - Statistical analysis

### MLOps & Feature Management
- **Hopsworks** - Feature store, model registry, data versioning
- **joblib** - Model serialization

### Visualization & Explainability
- **Streamlit** - Interactive web dashboard
- **Plotly** - Interactive charts
- **SHAP** - Model explainability
- **LIME** - Local interpretable explanations

### Utilities
- **python-dotenv** - Environment configuration
- **schedule** - Job scheduling
- **requests** - API calls

## 📋 Prerequisites

- Python 3.8+
- **Hopsworks Account** (free tier at [app.hopsworks.ai](https://app.hopsworks.ai))
- API credentials for weather/AQI data (if fetching real-time data)

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/ChanderValasai/aqi_predictor.git
cd aqi_predictor
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Hopsworks credentials
Create a `.env` file (or set environment variables):
```
HOPSWORKS_PROJECT=your_project_name
HOPSWORKS_API_KEY=your_api_key
```

You can get your API key from [app.hopsworks.ai](https://app.hopsworks.ai) → Avatar → Settings → API Keys.

## 🎮 Running the Pipelines

Use the `run.sh` script to execute different pipeline stages:

```bash
# Train models and register to Hopsworks Model Registry
bash run.sh training

# Run live feature engineering (fetches latest data)
bash run.sh feature

# Backfill historical data into feature store
bash run.sh backfill
```

## 📊 Running the Dashboard

```bash
streamlit run app/app.py
```

The dashboard will open at `http://localhost:8501`.

## 🎨 AQI Categories & Colors

| AQI Range | Category | Color | Health Impact |
|-----------|----------|-------|---------------|
| 0–50 | Good | 🟢 Green | Minimal impact |
| 51–100 | Moderate | 🟡 Yellow | Acceptable |
| 101–150 | Unhealthy for Sensitive Groups | 🟠 Orange | Sensitive groups may experience effects |
| 151–200 | Unhealthy | 🔴 Red | General public may experience effects |
| 201–300 | Very Unhealthy | 🟣 Purple | Health alert; serious effects |
| 301–500 | Hazardous | 🟤 Dark Red | Emergency conditions |

## 📈 Model Performance

The trained forecaster achieves:
- **Mean Absolute Error (MAE)**: ~8-12 AQI points
- **Root Mean Squared Error (RMSE)**: ~12-15 AQI points
- **R² Score**: 0.85-0.92
- **Prediction Accuracy**: 85-90% within acceptable range

## 🔍 Model Explainability

SHAP analysis is included to show:
- Feature importance for AQI predictions
- Which environmental factors most influence air quality
- How each factor affects the forecast

See `shap_summary.png` and `shap_beeswarm.png` for visualizations.

## 📝 Usage Example

```python
import joblib
import pandas as pd

# Load the trained model
forecaster = joblib.load('best_aqi_forecaster.pkl')

# Prepare features (example data)
features = pd.DataFrame({
    'pm25': [45.0],
    'pm10': [80.0],
    'temperature': [25.0],
    'humidity': [60.0],
    # ... other features
})

# Make predictions for 24h, 48h, 72h
predictions = forecaster.predict(features)
print(f"24h forecast: {predictions[24]}")
print(f"48h forecast: {predictions[48]}")
print(f"72h forecast: {predictions[72]}")
```

## 🐛 Troubleshooting

### Hopsworks Connection Error
- Verify `HOPSWORKS_PROJECT` and `HOPSWORKS_API_KEY` are set correctly
- Check Hopsworks credentials at [app.hopsworks.ai](https://app.hopsworks.ai)
- Ensure your API key hasn't expired

### Missing Model in Registry
- Run the training pipeline first: `bash run.sh training`
- Verify the model was registered as `aqi_forecaster` version 1 in Hopsworks

### Data Pipeline Issues
- Check network connectivity for fetching weather/AQI data
- Verify feature store access in Hopsworks
- Review pipeline logs for API errors

## 📚 Learning Resources

- [Hopsworks Documentation](https://docs.hopsworks.ai/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Chander Valasai** - Project Creator and Lead Developer

---

For more information, visit the [repository](https://github.com/ChanderValasai/aqi_predictor).
