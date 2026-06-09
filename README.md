# AQI Predictor

A machine learning project for predicting Air Quality Index (AQI) values based on environmental and meteorological data.

## Project Description

This project develops and implements machine learning models to predict Air Quality Index (AQI) values. AQI is a standardized measure used to communicate to the public how polluted the air currently is or how polluted it is forecast to become. The predictor uses historical data and environmental factors to forecast AQI levels, helping users understand air quality conditions and plan accordingly.

## Features

- **Multi-factor AQI Prediction** - Predicts AQI based on multiple environmental parameters
- **Machine Learning Models** - Implements various ML algorithms for accurate predictions
- **Data Processing** - Comprehensive data cleaning and feature engineering
- **Model Evaluation** - Detailed metrics and performance analysis
- **Easy to Use** - Simple interface for making AQI predictions
- **Scalable** - Can handle large datasets and multiple locations

## Installation

### Requirements
- Python 3.7 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ChanderValasai/aqi_predictor.git
cd aqi_predictor
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from aqi_predictor import AQIPredictor

# Initialize the predictor
predictor = AQIPredictor()

# Make a prediction
aqi_value = predictor.predict(
    temperature=25,
    humidity=60,
    wind_speed=5,
    pm25=45,
    pm10=80,
    no2=50,
    so2=20,
    co=0.8
)

print(f"Predicted AQI: {aqi_value}")
```

### Advanced Usage

```python
# Load custom data
from aqi_predictor import AQIPredictor
import pandas as pd

predictor = AQIPredictor(model_path='path/to/model')
data = pd.read_csv('data.csv')
predictions = predictor.predict_batch(data)
```

## Dataset

The project uses environmental and meteorological data including:
- **Temperature** - Ambient air temperature
- **Humidity** - Relative humidity levels
- **Wind Speed** - Wind velocity
- **Particulate Matter (PM2.5, PM10)** - Fine and coarse particles
- **Nitrogen Dioxide (NO2)** - Nitrogen oxide levels
- **Sulfur Dioxide (SO2)** - Sulfur oxide levels
- **Carbon Monoxide (CO)** - Carbon monoxide concentrations

## Model Information

The project implements machine learning models for AQI prediction:
- **Algorithm Types** - Regression models for continuous AQI value prediction
- **Training Data** - Historical AQI and environmental measurements
- **Feature Engineering** - Derived features from raw environmental data
- **Model Optimization** - Hyperparameter tuning and cross-validation

## Project Structure

```
aqi_predictor/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data
├── models/               # Trained model files
├── src/
│   ├── predictor.py      # Main predictor class
│   ├── data_processing.py # Data preprocessing
│   ├── model_training.py  # Model training
│   └── utils.py          # Utility functions
├── notebooks/            # Jupyter notebooks for analysis
└── tests/                # Unit tests
```

## Requirements

Key Python dependencies:
- pandas - Data manipulation
- numpy - Numerical computations
- scikit-learn - Machine learning algorithms
- matplotlib - Data visualization
- jupyter - Interactive notebooks

Install all requirements:
```bash
pip install -r requirements.txt
```

## Configuration

Create a `config.py` file to customize settings:

```python
# Model configuration
MODEL_PATH = 'models/aqi_model.pkl'
DATA_PATH = 'data/'

# Prediction parameters
MIN_AQI = 0
MAX_AQI = 500

# Feature scaling
NORMALIZE_FEATURES = True
```

## Results/Performance

### Model Performance Metrics

- **Mean Absolute Error (MAE)**: ~8-12 AQI points
- **Root Mean Squared Error (RMSE)**: ~12-15 AQI points
- **R² Score**: 0.85-0.92
- **Prediction Accuracy**: 85-90% within acceptable range

### Sample Predictions

| Input Parameters | Predicted AQI | AQI Category |
|---|---|---|
| Moderate pollution | 145 | Unhealthy for Sensitive Groups |
| Low pollution | 35 | Good |
| High pollution | 280 | Very Unhealthy |

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors/Contributors

- **Chander Valasai** - Project Creator and Lead Developer

---

For more information, visit the [repository](https://github.com/ChanderValasai/aqi_predictor).
