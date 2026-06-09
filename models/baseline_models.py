import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
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
    """
    Linear baseline with L2 regularization.
    RidgeCV auto-selects the best alpha via cross-validation instead of
    guessing alpha=1.0, which removes one source of underfitting/overfitting.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 500.0]))
    ])


def build_random_forest():
    """
    Strong non-linear baseline. Handles feature interactions well.
    Anti-overfitting changes vs. the original:
      - max_depth: 15 → 8   (shallower trees generalise better)
      - min_samples_leaf: 5 → 20  (each leaf needs more evidence)
      - max_features: 0.7 → 0.5  (more feature randomness = less variance)
    """
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        max_features=0.5,
        n_jobs=-1,
        random_state=42,
    )


def build_xgboost():
    """
    Gradient boosting. Early stopping is applied in MultiHorizonForecaster.fit()
    using the last 15 % of training as a validation set.
    Anti-overfitting changes vs. original:
      - max_depth: 6 → 4   (shallower)
      - min_child_weight: 5 → 20  (higher = less sensitive to noise)
      - reg_alpha: 0.1 → 1.0   (stronger L1)
      - reg_lambda: 1.0 → 5.0  (stronger L2)
      - n_estimators set high; early stopping finds the true optimum
    """
    return XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=20,
        reg_alpha=1.0,
        reg_lambda=5.0,
        eval_metric="rmse",
        early_stopping_rounds=50,   # XGBoost ≥ 2.0: constructor param, not fit()
        random_state=42,
        n_jobs=-1,
    )


def build_lightgbm():
    """
    Gradient boosting, leaf-wise growth. Early stopping applied in
    MultiHorizonForecaster.fit().
    Anti-overfitting changes vs. original:
      - max_depth: 6 → 4
      - num_leaves: 63 → 31  (2^4 − 1; keeps trees shallow)
      - min_child_samples: 20 → 50  (more data per leaf)
      - reg_alpha: 0.1 → 1.0
      - reg_lambda: 1.0 → 5.0
    """
    return LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=50,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
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
        self.models        = {}
        self.base_model_fn = base_model_fn
        self.horizons      = [24, 48, 72]
        self.feature_names = None   # set during fit(); used to avoid name warnings

    def _as_df(self, X):
        """Wrap X in a DataFrame if feature names are available."""
        if self.feature_names is not None and not isinstance(X, pd.DataFrame):
            return pd.DataFrame(X, columns=self.feature_names)
        return X

    def fit(self, X, y_dict, val_fraction=0.15, feature_names=None):
        """
        X            : numpy array or DataFrame of shape (n_samples, n_features)
        y_dict       : {"24h": array, "48h": array, "72h": array}
        val_fraction : last fraction of training rows used as an *internal*
                       validation set for early stopping (XGBoost / LightGBM).
                       Ridge and RandomForest always train on the full X.
        feature_names: list of column names — prevents LightGBM/XGBoost from
                       emitting "X does not have valid feature names" at predict time.
        """
        if feature_names is not None:
            self.feature_names = list(feature_names)

        n_val = max(50, int(len(X) * val_fraction))

        # Boosting models get a DataFrame so names are stored consistently
        X_df     = self._as_df(X)
        X_df_tr  = X_df.iloc[:-n_val] if isinstance(X_df, pd.DataFrame) else X_df[:-n_val]
        X_df_val = X_df.iloc[-n_val:]  if isinstance(X_df, pd.DataFrame) else X_df[-n_val:]

        for h in self.horizons:
            key   = f"{h}h"
            y_tr  = y_dict[key][:-n_val]
            y_val = y_dict[key][-n_val:]

            print(f"\n🏋️  Training {h}h horizon model...")
            model      = self.base_model_fn()
            model_name = type(model).__name__

            if model_name == "XGBRegressor":
                model.fit(
                    X_df_tr, y_tr,
                    eval_set=[(X_df_val, y_val)],
                    verbose=False,
                )
                best_r = model.best_iteration
                n_trees = model.get_booster().num_boosted_rounds()
                print(f"   ↳ best round: {best_r}  |  trees trained: {n_trees}"
                      f"  |  stopped {n_trees - best_r} rounds after best")

            elif model_name == "LGBMRegressor":
                model.fit(
                    X_df_tr, y_tr,
                    eval_set=[(X_df_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )
                best_r = model.best_iteration_
                # LightGBM trims the booster to best_iteration_ after early stopping,
                # so n_estimators_ == best_iteration_.  Actual rounds run = best + 50.
                print(f"   ↳ best round: {best_r}  |  early-stopped after ~{best_r + 50} rounds")

            else:
                # Ridge (Pipeline) and RandomForest: fit with DataFrame so that
                # predict() (which also receives a DataFrame via _as_df) stays consistent
                model.fit(X_df, y_dict[key])

            self.models[key] = model
        return self

    def predict(self, X):
        X_in = self._as_df(X)
        return {h: self.models[f"{h}h"].predict(X_in) for h in [24, 48, 72]}

    def evaluate_all(self, X_test, y_test_dict):
        X_in    = self._as_df(X_test)
        results = []
        for h in self.horizons:
            key    = f"{h}h"
            y_pred = self.models[key].predict(X_in)
            metrics = evaluate(y_test_dict[key], y_pred, f"XGBoost (+{key})")
            results.append(metrics)
        return pd.DataFrame(results)
