import os
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
    df = df.sort_values("timestamp").dropna()

    # Define features and multi-horizon targets
    target_cols = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
    drop_cols   = ["timestamp"] + target_cols
    feature_cols = [c for c in df.columns if c not in drop_cols]

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
