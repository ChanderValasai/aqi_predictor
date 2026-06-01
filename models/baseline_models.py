import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from model_utils import evaluate, explain_with_shap, save_model

# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_training_data(feature_view, target_col="target_aqi_24h"):
    """
    Fetch features from Hopsworks Feature View and prepare X, y.
    A Feature View joins feature groups and provides a training dataset.
    """
    X, y = feature_view.get_training_data(1)
    y = X.pop(target_col)
    # Drop other target columns (48h, 72h) from features
    for col in ["target_aqi_48h", "target_aqi_72h"]:
        if col in X.columns:
            X.drop(columns=[col], inplace=True)
    return X, y


# ─── TIME-SERIES CROSS-VALIDATION ────────────────────────────────────────────
# CRITICAL: Never use random splits for time-series data.
# Always use TimeSeriesSplit to respect temporal order.

def time_series_cv(model, X, y, n_splits=5):
    """
    Walk-forward validation. Each fold trains on the past, tests on the future.
    
    Example with n_splits=5 and 1000 rows:
      Fold 1: Train [0..166]   Test [167..332]
      Fold 2: Train [0..332]   Test [333..499]
      Fold 3: Train [0..499]   Test [500..666]
      Fold 4: Train [0..666]   Test [667..832]
      Fold 5: Train [0..832]   Test [833..999]
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        fold_scores.append(rmse)
        print(f"  Fold {fold}: RMSE = {rmse:.3f}")

    print(f"  Avg CV RMSE: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")
    return fold_scores


# ─── MODEL DEFINITIONS ────────────────────────────────────────────────────────

def build_ridge():
    """Linear baseline with L2 regularization."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0))
    ])


def build_random_forest():
    """
    Strong non-linear baseline. Handles feature interactions well.
    Key hyperparameters:
      - n_estimators: more trees = better but slower (100-500)
      - max_depth: controls overfitting (None = fully grown, risky for AQI)
      - min_samples_leaf: minimum samples per leaf, improves generalization
    """
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        max_features=0.7,        # use 70% of features at each split
        n_jobs=-1,
        random_state=42,
    )


def build_xgboost():
    """
    Usually the best performer on tabular time-series data.
    Key hyperparameters:
      - learning_rate (eta): smaller = better generalization (try 0.01-0.3)
      - n_estimators: number of boosting rounds (use early stopping)
      - max_depth: depth of trees (3-8 for tabular data)
      - subsample: row sampling per tree (0.6-1.0)
      - colsample_bytree: feature sampling per tree (0.6-1.0)
    """
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,           # L1 regularization
        reg_lambda=1.0,          # L2 regularization
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )


def build_lightgbm():
    """
    Faster than XGBoost, often comparable accuracy.
    Better for large datasets (>100k rows).
    """
    return LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,           # 2^max_depth - 1 is a good rule of thumb
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )


# ─── HYPERPARAMETER TUNING ────────────────────────────────────────────────────

def tune_xgboost(X_train, y_train):
    """
    RandomizedSearchCV with TimeSeriesSplit — find best XGBoost params.
    Use this after your first round of results to squeeze out more performance.
    """
    param_dist = {
        "n_estimators":    [200, 300, 500],
        "learning_rate":   [0.01, 0.05, 0.1, 0.2],
        "max_depth":       [3, 4, 5, 6, 7, 8],
        "subsample":       [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree":[0.6, 0.7, 0.8, 1.0],
        "min_child_weight":[1, 3, 5, 10],
        "reg_alpha":       [0, 0.1, 0.5, 1.0],
        "reg_lambda":      [0.5, 1.0, 2.0, 5.0],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=50,                     # try 50 random combinations
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        verbose=1,
        random_state=42,
    )
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV RMSE: {-search.best_score_:.3f}")
    return search.best_estimator_


# ─── MULTI-HORIZON FORECASTING ────────────────────────────────────────────────
# Strategy: train separate models for each forecast horizon (24h, 48h, 72h)

class MultiHorizonForecaster:
    """
    Trains one model per forecast horizon (direct multi-step approach).
    More accurate than recursive (chained) prediction for AQI.
    """
    def __init__(self, base_model_fn=build_xgboost):
        self.models = {}
        self.base_model_fn = base_model_fn
        self.horizons = [24, 48, 72]

    def fit(self, X, y_dict):
        """y_dict = {"24h": Series, "48h": Series, "72h": Series}"""
        for h in self.horizons:
            key = f"{h}h"
            print(f"\n🏋️  Training {h}h horizon model...")
            model = self.base_model_fn()
            model.fit(X, y_dict[key])
            self.models[key] = model
        return self

    def predict(self, X):
        return {h: self.models[f"{h}h"].predict(X) for h in [24, 48, 72]}

    def evaluate_all(self, X_test, y_test_dict):
        results = []
        for h in self.horizons:
            key = f"{h}h"
            y_pred = self.models[key].predict(X_test)
            metrics = evaluate(y_test_dict[key], y_pred, f"XGBoost (+{key})")
            results.append(metrics)
        return pd.DataFrame(results)
