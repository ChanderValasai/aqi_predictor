
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import shap
import joblib, os

# ─── METRICS ─────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, model_name="Model") -> dict:
    """Compute and print RMSE, MAE, R²."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    metrics = {"model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE%": mape}
    print(f"\n{'='*45}")
    print(f"  {model_name}")
    print(f"  RMSE : {rmse:.3f}")
    print(f"  MAE  : {mae:.3f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"{'='*45}")
    return metrics


def persistence_baseline(y_true, lag=1):
    """Naive persistence: predict t+n = t."""
    y_pred = np.roll(y_true, lag)
    y_pred[:lag] = y_true[:lag]
    return evaluate(y_true, y_pred, f"Persistence Baseline (lag={lag}h)")


# ─── SHAP EXPLANATIONS ───────────────────────────────────────────────────────

def explain_with_shap(model, X_train, X_test, feature_names, model_type="tree"):
    """
    Generate SHAP values and summary plot.
    model_type: 'tree' for RF/XGBoost, 'linear' for Ridge, 'kernel' for DL
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X_train)
    else:
        explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))

    shap_values = explainer.shap_values(X_test)

    # Summary bar plot (global feature importance)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150)
    plt.close()

    # Beeswarm plot (full distribution)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("shap_beeswarm.png", dpi=150)
    plt.close()

    return shap_values


# ─── MODEL PERSISTENCE ───────────────────────────────────────────────────────

def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"💾 Model saved → {path}")


def load_model(path: str):
    return joblib.load(path)
