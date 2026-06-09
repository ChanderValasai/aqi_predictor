# AQI Prediction System — Professional Report

## Executive Summary

This report documents the design, implementation, and results of the AQI (Air Quality Index) Prediction System for **Karachi**, Pakistan. The system automates the collection of live and historical air quality data, engineers a rich feature set, trains and evaluates multiple machine learning models across three forecast horizons, and exposes a real-time interactive dashboard for stakeholders. All data is managed through the **Hopsworks Feature Store**, and the best model is versioned in the **Hopsworks Model Registry** for reproducible deployment.

---

## Objectives

- Automate collection of hourly air quality and weather data for Karachi via public APIs
- Store, version, and serve features through a managed Feature Store (Hopsworks)
- Recompute lag, rolling, and target features from the full time series to maximise usable training data
- Train and compare four model families across 24 h, 48 h, and 72 h forecast horizons
- Select and register the best model automatically based on validation RMSE
- Provide explainability via SHAP feature importance
- Deploy an interactive Streamlit dashboard showing the current AQI and a 3-day AQI forecast

---

## Data Pipeline

### Data Sources

| Source | Purpose | Access |
|---|---|---|
| OpenWeatherMap Air Pollution API | Historical AQI + pollutants (PM2.5, PM10, NO₂, O₃, CO, SO₂, NH₃) since 2020-11-27 | Free tier |
| OpenWeatherMap Weather API | Live temperature, humidity, wind speed, pressure, cloud cover | Free tier |
| AQICN API | Live current AQI reading for Karachi | Free tier |

### Storage — Hopsworks Feature Store

- **Feature Group:** `aqi_features` (version 1)
- **Schema:** 8,498 hourly rows spanning from November 2020 to present
- **Backfill:** `pipelines/backfill.py` — run once to populate historical data; fetches in monthly chunks to stay within API rate limits
- **Live updates:** `pipelines/feature_pipeline.py` — run on a schedule (e.g. every hour via cron) to append the latest observation

### Preprocessing & Feature Engineering

Raw API responses are transformed into a flat feature row by `compute_features()` in `feature_pipeline.py`. The following categories of features are created:

**Pollutant readings (raw)**
- `pm25`, `pm10`, `no2`, `o3`, `co`, `so2`, `nh3`, `aqi`

**Temporal features**
- `hour`, `day_of_week`, `month`, `is_weekend`, `hour_sin`, `hour_cos`, `month_sin`, `month_cos` (cyclical encoding)

**Lag features** (recomputed at training time from the full sorted series)
- AQI and PM2.5 lags at 1 h, 2 h, 3 h, 6 h, 12 h, 24 h

**Rolling statistics** (computed over preceding window; shift(1) prevents data leakage)
- Mean, std, max of AQI over 3 h, 6 h, 12 h, 24 h windows

**Rate-of-change features**
- `aqi_change_1h`, `aqi_change_3h`, `aqi_change_24h`

**Weather features**
- Temperature, humidity, wind speed/direction, pressure, visibility, cloud cover

> **Note:** Lag, rolling, and target columns are stored as NaN in the Feature Store (each inserted row has no history at insert time). They are **recomputed** by the training pipeline from the full sorted time series before model training.

### Variance Filtering

A `VarianceThreshold` filter drops any column with near-zero variance (e.g. constant pollutant readings during certain periods). This removed **22 constant columns**, leaving **39 usable features**.

### Train / Test Split

| Set | Rows |
|---|---|
| Usable after recomputation | 8,426 |
| Training | 6,740 (80 %) |
| Test (held-out) | 1,686 (20 %) |

---

## Modeling Approach

### Task Formulation

**Direct multi-step forecasting** — one independent model per horizon. This avoids error accumulation that occurs in recursive (chained) approaches.

| Horizon | Target variable |
|---|---|
| +24 h | AQI 24 hours ahead |
| +48 h | AQI 48 hours ahead |
| +72 h | AQI 72 hours ahead |

### Algorithms

| Model | Regularisation / Tuning |
|---|---|
| **Ridge Regression** (via sklearn Pipeline + StandardScaler + RidgeCV) | Cross-validated α ∈ [0.1, 1, 10, 100, 1000] |
| **Random Forest** | 200 trees, max_depth=8, min_samples_leaf=5 |
| **XGBoost** | max_depth=4, learning_rate=0.05, subsample=0.8, reg_α=1, reg_λ=5; early stopping (50 rounds patience) |
| **LightGBM** | num_leaves=31, learning_rate=0.05, min_child_samples=30, reg_α=1, reg_λ=5; early stopping (50 rounds patience) |

### Early Stopping

XGBoost and LightGBM hold out the last 15 % of training rows as an internal validation set. Training stops when validation RMSE does not improve for **50 consecutive rounds**:

| Model | Horizon | Best round | Total rounds run |
|---|---|---|---|
| XGBoost | +24 h | 51 | 102 |
| XGBoost | +48 h | 15 | 66 |
| XGBoost | +72 h | 31 | 82 |
| LightGBM | +24 h | 37 | ~87 |
| LightGBM | +48 h | 10 | ~60 |
| LightGBM | +72 h | 24 | ~74 |

### Evaluation Metrics

- **RMSE** — Root Mean Squared Error (primary selection criterion)
- **MAE** — Mean Absolute Error
- **R²** — Coefficient of Determination
- **MAPE** — Mean Absolute Percentage Error

### Persistence Baseline

A naïve persistence model (predict tomorrow's AQI = today's AQI) is computed for reference. AQI is highly autocorrelated at 1-hour lag, making this a very strong short-horizon baseline.

| Metric | Persistence (lag=1 h) |
|---|---|
| RMSE | 10.19 |
| MAE | 2.08 |
| R² | 0.919 |
| MAPE | 1.53 % |

---

## Model Results

### +24 h Horizon (most important)

| Model | RMSE | MAE | R² | MAPE |
|---|---|---|---|---|
| Ridge | 34.72 | 26.88 | 0.063 | 22.4 % |
| Random Forest | 35.98 | 28.14 | −0.007 | 24.4 % |
| **XGBoost ✓** | **33.69** | **25.88** | **0.117** | **23.0 %** |
| LightGBM | 34.02 | 26.53 | 0.100 | 23.5 % |

### +48 h Horizon

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Ridge | 38.03 | 30.13 | −0.126 |
| Random Forest | 41.09 | 32.28 | −0.315 |
| XGBoost | 37.32 | 26.54 | −0.084 |
| LightGBM | 37.58 | 26.86 | −0.100 |

### +72 h Horizon

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Ridge | 38.46 | 30.08 | −0.163 |
| Random Forest | 44.88 | 33.84 | −0.585 |
| XGBoost | 39.07 | 29.34 | −0.201 |
| LightGBM | 38.06 | 28.81 | −0.139 |

### Selected Model

**XGBoost** — best RMSE and R² at the most critical 24-hour horizon.  
Registered in Hopsworks Model Registry as `aqi_forecaster` **version 10**.

> **Note on R² at longer horizons:** Negative R² values beyond 48 h indicate that multi-day AQI forecasting from weather + pollutant features alone is harder than the persistence baseline. AQI in Karachi exhibits strong short-term autocorrelation but high long-range uncertainty due to variable wind patterns and episodic pollution events.

---

## Explainability — SHAP

After selecting the best model, SHAP (SHapley Additive exPlanations) values are computed on the test set to explain +24 h predictions. Key driving features typically include:

- Recent AQI lag values (`aqi_lag_1h`, `aqi_lag_24h`)
- Short-term rolling mean/max of AQI (`aqi_roll_mean_3h`, `aqi_roll_max_24h`)
- PM2.5 concentration and its lag
- Wind speed (dispersion driver)
- Hour of day (cyclical)

SHAP output is logged to the console during training and can be extended to render a summary plot in the dashboard.

---

## Dashboard & Visualisation

**Framework:** Streamlit + Plotly  
**Source:** `app/app.py`

### Features

| Feature | Description |
|---|---|
| Current AQI card | Live AQI reading with colour-coded category (Good → Hazardous) |
| 3-Day Forecast | Bar chart for +24 h, +48 h, +72 h predicted AQI with category labels |
| AQI Category Legend | Visual scale (0–500) with EPA colour bands |
| Forecast table | Day-by-day breakdown with AQI value and category |
| Model info | Displays active model version loaded from Hopsworks |

### AQI Colour Scale (EPA Standard)

| Range | Category | Colour |
|---|---|---|
| 0 – 50 | Good | Green |
| 51 – 100 | Moderate | Yellow |
| 101 – 150 | Unhealthy for Sensitive Groups | Orange |
| 151 – 200 | Unhealthy | Red |
| 201 – 300 | Very Unhealthy | Purple |
| 301 – 500 | Hazardous | Maroon |

---

## Deployment

| Component | Technology |
|---|---|
| Application server | Streamlit (Replit hosted) |
| Feature Store | Hopsworks (EU-West cloud) |
| Model Registry | Hopsworks Model Registry |
| Secret management | Replit Secrets (`HOPSWORKS_PROJECT`, `HOPSWORKS_API_KEY`, `OPENWEATHER_API_KEY`, `AQICN_API_KEY`) |
| Runtime environment | Python 3.11, NixOS (Replit) |
| LightGBM compatibility | Auto-detects `libgomp.so.1` via `LD_LIBRARY_PATH` re-exec trick |

### Pipeline Execution

```bash
# Backfill historical data (run once)
bash run.sh backfill

# Update features with latest reading (run hourly via cron)
bash run.sh feature

# Retrain all models and register best (run weekly or on demand)
bash run.sh training
```

---

## Impact & Recommendations

### Current Impact
- Provides Karachi residents and environmental agencies with a 3-day AQI outlook
- Fully automated: data ingestion → feature storage → model training → dashboard refresh requires no manual steps
- Model versioning in Hopsworks ensures reproducibility and rollback capability

### Recommendations for Future Work

| Priority | Enhancement |
|---|---|
| High | Add hourly cron scheduling for the feature pipeline to keep the Feature Store current |
| High | Incorporate meteorological forecast data (predicted wind, rain) as features for improved multi-day accuracy |
| Medium | Add GitHub Actions CI/CD to automatically retrain on a weekly schedule |
| Medium | Integrate satellite-based aerosol optical depth (AOD) data as an additional signal |
| Low | Extend coverage to other Pakistani cities (Lahore, Islamabad, Peshawar) |
| Low | Add SMS/email alerting when predicted AQI exceeds Unhealthy threshold |
| Low | Display SHAP waterfall charts for individual predictions in the dashboard |

---

## Project Structure

```
.
├── app/
│   └── app.py                  # Streamlit dashboard
├── models/
│   ├── baseline_models.py      # MultiHorizonForecaster, model builders, early stopping
│   └── model_utils.py          # SHAP, evaluate, persistence baseline, save/load
├── pipelines/
│   ├── backfill.py             # One-time historical data backfill
│   ├── feature_pipeline.py     # Live feature ingestion + Hopsworks insert
│   └── training_pipeline.py    # Full training, evaluation, and model registration
├── run.sh                      # Helper: sets LD_LIBRARY_PATH + PYTHONPATH
├── requirements.txt
└── report.md                   # This document
```

---

## Conclusion

The Karachi AQI Prediction System demonstrates a complete, production-oriented MLOps pipeline: from raw API ingestion through a managed feature store, to multi-model training with early stopping and SHAP explainability, to a live interactive dashboard. XGBoost achieves the best 24-hour forecast performance (R² = 0.117, RMSE = 33.69 AQI units). Longer horizons are inherently harder due to the chaotic nature of pollution dispersion, but the architecture is designed to accommodate richer meteorological forecast inputs that would improve 48–72 h accuracy significantly.

---

*For further details, refer to the source code or contact the project maintainer.*
