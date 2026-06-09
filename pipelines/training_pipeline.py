import os
import sys
import subprocess

# ── libgomp auto-fix (LightGBM needs it; re-exec with LD_LIBRARY_PATH if missing) ──
def _ensure_libgomp():
    try:
        import ctypes
        ctypes.cdll.LoadLibrary("libgomp.so.1")
    except OSError:
        result = subprocess.run(
            ["gcc", "--print-file-name=libgomp.so.1"],
            capture_output=True, text=True
        )
        gomp_dir = os.path.dirname(result.stdout.strip())
        current = os.environ.get("LD_LIBRARY_PATH", "")
        if gomp_dir not in current:
            os.environ["LD_LIBRARY_PATH"] = f"{gomp_dir}:{current}"
            os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_libgomp()

# ── sys.path: allow running as `python pipelines/training_pipeline.py` ──────────
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_repo_root, "models"), os.path.join(_repo_root, "pipelines")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import numpy as np
import hopsworks
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from baseline_models import (
    build_xgboost, build_lightgbm, build_random_forest,
    build_ridge, MultiHorizonForecaster, time_series_cv
)
from model_utils import evaluate, persistence_baseline, explain_with_shap, save_model

load_dotenv()


def run_training():
    # ── 1. Connect to Hopsworks ──────────────────────────────────────────────
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    )
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # ── 2. Fetch Training Data ───────────────────────────────────────────────
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"📥 Raw rows from Feature Store: {len(df)}")

    # ── 2a. Recompute lag/rolling/target columns from the full time series ───
    # The backfill pipeline stored NaN for these columns because individual
    # hourly snapshots have no history. We recompute them here from the
    # sorted series so that all ~8,400 historical rows become usable.

    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"aqi_lag_{lag}h"]  = df["aqi"].shift(lag)
        df[f"pm25_lag_{lag}h"] = df["pm25"].shift(lag)

    for window in [3, 6, 12, 24]:
        # shift(1) avoids leaking the current value into the rolling window
        rolled = df["aqi"].shift(1).rolling(window)
        df[f"aqi_roll_mean_{window}h"] = rolled.mean()
        df[f"aqi_roll_std_{window}h"]  = rolled.std()
        df[f"aqi_roll_max_{window}h"]  = rolled.max()

    df["aqi_change_1h"]  = df["aqi"].diff(1)
    df["aqi_change_3h"]  = df["aqi"].diff(3)
    df["aqi_change_24h"] = df["aqi"].diff(24)

    # Targets: future AQI values
    df["target_aqi_24h"] = df["aqi"].shift(-24)
    df["target_aqi_48h"] = df["aqi"].shift(-48)
    df["target_aqi_72h"] = df["aqi"].shift(-72)

    # ── 2b. Build feature/target arrays ─────────────────────────────────────
    target_cols  = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
    drop_cols    = ["timestamp", "weather_main"] + target_cols
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Drop rows where targets are NaN (last 72 rows) or core inputs are NaN
    required_cols = target_cols + ["aqi", "pm25"]
    df = df.dropna(subset=required_cols)

    # Fill any remaining NaNs (e.g. first few rows missing early lags, weather cols)
    df[feature_cols] = (
        df[feature_cols]
        .fillna(df[feature_cols].median(numeric_only=True))
        .fillna(0)
    )

    print(f"✅ Usable rows after recomputing features: {len(df)}")

    X = df[feature_cols].values
    y = {
        "24h": df["target_aqi_24h"].values,
        "48h": df["target_aqi_48h"].values,
        "72h": df["target_aqi_72h"].values,
    }

    # ── 3. Temporal Train/Test Split ─────────────────────────────────────────
    # Use last 20% of time for testing (never shuffle time series!)
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train = {h: v[:split_idx] for h, v in y.items()}
    y_test  = {h: v[split_idx:] for h, v in y.items()}

    print(f"📊 Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # ── 4. Baseline: Persistence ─────────────────────────────────────────────
    print("\n📐 Persistence Baseline (beat this!)")
    persistence_baseline(y_test["24h"], lag=1)

    # ── 5. Train All Models ───────────────────────────────────────────────────
    results = []
    trained_models = {}

    model_builders = {
        "Ridge":        build_ridge,
        "RandomForest": build_random_forest,
        "XGBoost":      build_xgboost,
        "LightGBM":     build_lightgbm,
    }

    for name, builder in model_builders.items():
        print(f"\n🚀 Training {name}...")
        forecaster = MultiHorizonForecaster(base_model_fn=builder)
        forecaster.fit(X_train, y_train)
        metrics_df = forecaster.evaluate_all(X_test, y_test)
        metrics_df["model_family"] = name
        results.append(metrics_df)
        trained_models[name] = forecaster

    # ── 6. Select Best Model ──────────────────────────────────────────────────
    all_results = pd.concat(results, ignore_index=True)
    # Pick best based on 24h RMSE (most important horizon)
    best_24h = all_results[all_results["model"] == "XGBoost (+24h)"].sort_values("RMSE")
    best_model_name = best_24h.iloc[0]["model_family"] if len(best_24h) else "XGBoost"
    best_forecaster = trained_models[best_model_name]
    print(f"\n🏆 Best model: {best_model_name}")

    # ── 7. SHAP Explainability ───────────────────────────────────────────────
    print("\n🔍 Computing SHAP values...")
    explain_with_shap(
        model=best_forecaster.models["24h"],
        X_train=X_train, X_test=X_test[:200],
        feature_names=feature_cols,
        model_type="tree"
    )

    # ── 8. Register Best Model in Hopsworks ──────────────────────────────────
    save_model(best_forecaster, "best_aqi_forecaster.pkl")

    model_meta = mr.python.create_model(
        name="aqi_forecaster",
        metrics={
            "rmse_24h": float(best_24h.iloc[0]["RMSE"]),
            "mae_24h":  float(best_24h.iloc[0]["MAE"]),
            "r2_24h":   float(best_24h.iloc[0]["R2"]),
        },
        description=f"Multi-horizon AQI forecaster ({best_model_name})",
        input_example=X_test[:1],
    )
    model_meta.save("best_aqi_forecaster.pkl")
    print(f"✅ Model registered in Hopsworks Model Registry (v{model_meta.version})")

    return best_forecaster, all_results


if __name__ == "__main__":
    run_training()
