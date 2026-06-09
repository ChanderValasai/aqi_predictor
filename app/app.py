
import os
import sys

# Ensure models/ and pipelines/ are importable on Streamlit Community Cloud
# (on Replit, PYTHONPATH is set in the workflow; on Cloud it is not)
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_repo_root, "models"), os.path.join(_repo_root, "pipelines")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import hopsworks
import joblib
import shap
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Predictor — Karachi",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  SECRETS
# ─────────────────────────────────────────────────────────────────────────────
def get_secret(key):
    """Read from Replit Secrets (env vars) first, then fall back to st.secrets."""
    value = os.environ.get(key, "").strip()
    if value:
        return value
    try:
        value = st.secrets.get(key, "").strip()
        return value if value else None
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
#  AQI COLOUR SCALE
# ─────────────────────────────────────────────────────────────────────────────
AQI_LEVELS = [
    (0,   50,  "Good",              "#00e400"),
    (51,  100, "Moderate",          "#ffff00"),
    (101, 150, "Unhealthy for SGs", "#ff7e00"),
    (151, 200, "Unhealthy",         "#ff0000"),
    (201, 300, "Very Unhealthy",    "#8f3f97"),
    (301, 500, "Hazardous",         "#7e0023"),
]

AQI_BAR_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy for SGs": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023",
}


def get_aqi_category(aqi_val):
    for lo, hi, label, color in AQI_LEVELS:
        if lo <= aqi_val <= hi:
            return label, color
    return "Hazardous", "#7e0023"


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(ttl=1800, show_spinner=False)
def load_model_and_data():
    """Load model + last 2000 rows from Hopsworks (cached 30 min)."""
    project = hopsworks.login(
        project=get_secret("HOPSWORKS_PROJECT"),
        api_key_value=get_secret("HOPSWORKS_API_KEY"),
    )
    mr = project.get_model_registry()
    # Load latest model version (v12 is the newest trained model)
    all_models = mr.get_models("aqi_forecaster")
    model_meta = max(all_models, key=lambda m: m.version) if all_models else None
    if model_meta is None:
        raise RuntimeError(
            "No model named **aqi_forecaster** found in your Hopsworks Model Registry. "
            "Please run the training pipeline first."
        )
    model_dir = model_meta.download()
    forecaster = joblib.load(f"{model_dir}/best_aqi_forecaster.pkl")

    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return forecaster, df


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🌫️ AQI Predictor")
st.sidebar.markdown("**City:** Karachi, Pakistan")
st.sidebar.markdown("**Lat / Lon:** 24.86°N, 67.01°E")
st.sidebar.markdown("**Population:** ~15 million")
st.sidebar.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Live Dashboard",
    "🔍 EDA Analysis",
    "🏆 Model Performance",
    "💡 SHAP Explainability",
])

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — LIVE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    try:
        forecaster, df = load_model_and_data()
        df = df.tail(2000)  # last 2000 hours for display

        # ── Current AQI + City Details ──────────────────────────────────────
        current_aqi = int(df["aqi"].iloc[-1])
        label, color = get_aqi_category(current_aqi)

        st.markdown("## 📊 Current Conditions")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("AQI", current_aqi, label)
        with c2:
            st.metric("PM2.5", f"{df['pm25'].iloc[-1]:.1f}")
        with c3:
            pm10_val = df['pm10'].iloc[-1]
            st.metric("PM10", f"{pm10_val:.1f}" if pd.notna(pm10_val) else "N/A")
        with c4:
            st.metric("Temp", f"{df['temperature'].iloc[-1]:.1f} °C")

        st.markdown(
            f'<div style="background:{color};color:#000;padding:12px;border-radius:8px;'
            f'text-align:center;font-weight:bold;font-size:1.2em;">'
            f'Air Quality: {label}</div>',
            unsafe_allow_html=True,
        )

        if current_aqi > 150:
            st.error(
                f"⚠️ **HAZARDOUS ALERT** — AQI is {current_aqi}. "
                "Avoid outdoor activities."
            )

        st.divider()

        # ── More City Details ───────────────────────────────────────────────
        st.markdown("### 🏙️ City Details")
        cd1, cd2, cd3, cd4 = st.columns(4)
        with cd1:
            st.metric("Humidity", f"{df['humidity'].iloc[-1]:.0f}%")
        with cd2:
            st.metric("Wind Speed", f"{df['wind_speed'].iloc[-1]:.1f} m/s")
        with cd3:
            st.metric("Pressure", f"{df['pressure'].iloc[-1]:.0f} hPa")
        with cd4:
            st.metric("Visibility", f"{df['visibility'].iloc[-1]:.1f} km")

        st.markdown(
            f"**Last updated:** {df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M UTC')}  "
            f"| **Records in store:** {len(df)} hours"
        )

        st.divider()

        # ── 3-Day Forecast ──────────────────────────────────────────────────
        st.markdown("## 📅 3-Day AQI Forecast")
        drop_cols = [
            "timestamp", "weather_main",
            "target_aqi_24h", "target_aqi_48h", "target_aqi_72h",
        ]
        feature_cols = [c for c in df.columns if c not in drop_cols]
        X_df = df[feature_cols].copy()
        X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)
        X_latest = X_df.iloc[[-1]].values

        predictions = forecaster.predict(X_latest)
        now = datetime.now()
        forecast_dates = [
            (now + timedelta(hours=24)).strftime("%b %d"),
            (now + timedelta(hours=48)).strftime("%b %d"),
            (now + timedelta(hours=72)).strftime("%b %d"),
        ]

        fc1, fc2, fc3 = st.columns(3)
        for col, h, date in zip([fc1, fc2, fc3], [24, 48, 72], forecast_dates):
            pred_val = int(np.asarray(predictions[h]).flat[0])
            lbl, clr = get_aqi_category(pred_val)
            with col:
                st.markdown(
                    f'<div style="border:3px solid {clr};border-radius:10px;'
                    f'padding:18px;text-align:center;background:#1a1a2e;">'
                    f'<b style="font-size:1.1em">{date}</b><br>'
                    f'<span style="font-size:2.5em;font-weight:bold;color:{clr};">'
                    f'{pred_val}</span><br>'
                    f'<span style="font-size:0.9em">{lbl}</span></div>',
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── Historical AQI Chart ────────────────────────────────────────────
        st.markdown("### 📈 Historical AQI (Last 7 Days)")
        fig = px.line(
            df.tail(168), x="timestamp", y="aqi",
            labels={"aqi": "AQI", "timestamp": "Time"},
            template="plotly_dark",
        )
        fig.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="Unhealthy")
        fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Moderate")
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')

        # ── Pollutant Breakdown ───────────────────────────────────────────
        st.markdown("### 🧪 Pollutant Levels")
        pollutants = ["pm25", "pm10", "o3", "no2", "so2", "co"]
        latest_vals = [float(df[p].iloc[-1]) if pd.notna(df[p].iloc[-1]) else 0 for p in pollutants]
        bar_fig = px.bar(
            x=pollutants, y=latest_vals,
            labels={"x": "Pollutant", "y": "Concentration (µg/m³)"},
            color=latest_vals,
            color_continuous_scale="RdYlGn_r",
            template="plotly_dark",
        )
        bar_fig.update_layout(height=350)
        st.plotly_chart(bar_fig, width='stretch')

    except Exception as e:
        st.error(f"❌ Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — EDA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    try:
        _, df = load_model_and_data()
        df = df.tail(2000)
        st.markdown("## 🔍 Exploratory Data Analysis")
        st.markdown("Insights from the last 2,000 hourly records in the Feature Store.")

        # ── Row 1: AQI Distribution + Category Pie ─────────────────────────
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            st.markdown("### AQI Distribution")
            hist_fig = px.histogram(
                df, x="aqi", nbins=50,
                color_discrete_sequence=["#636EFA"],
                template="plotly_dark",
            )
            hist_fig.add_vline(x=df["aqi"].mean(), line_dash="dash", line_color="red",
                               annotation_text=f"Mean: {df['aqi'].mean():.1f}")
            hist_fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(hist_fig, use_allow_html=True)

        with r1c2:
            st.markdown("### AQI Categories")
            categories = []
            for val in df["aqi"]:
                cat, _ = get_aqi_category(val)
                categories.append(cat)
            cat_counts = pd.Series(categories).value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            cat_counts["Color"] = cat_counts["Category"].map(AQI_BAR_COLORS)
            pie_fig = px.pie(
                cat_counts, values="Count", names="Category",
                color="Category", color_discrete_map=AQI_BAR_COLORS,
                template="plotly_dark",
            )
            pie_fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(pie_fig, width='stretch')

        # ── Row 2: Hourly Pattern + Monthly Trend ──────────────────────────
        r2c1, r2c2 = st.columns(2)

        with r2c1:
            st.markdown("### AQI by Hour of Day")
            df["hour"] = df["timestamp"].dt.hour
            hourly = df.groupby("hour")["aqi"].mean().reset_index()
            h_fig = px.line(
                hourly, x="hour", y="aqi",
                markers=True, template="plotly_dark",
                labels={"aqi": "Avg AQI", "hour": "Hour of Day"},
            )
            h_fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(h_fig, width='stretch')

        with r2c2:
            st.markdown("### AQI by Month")
            df["month_name"] = df["timestamp"].dt.strftime("%b")
            monthly = df.groupby("month_name")["aqi"].mean().reindex(
                ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            ).reset_index()
            m_fig = px.bar(
                monthly, x="month_name", y="aqi",
                color="aqi", color_continuous_scale="RdYlGn_r",
                template="plotly_dark",
                labels={"aqi": "Avg AQI", "month_name": "Month"},
            )
            m_fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(m_fig, width='stretch')

        # ── Row 3: Correlation Heatmap ─────────────────────────────────────
        st.markdown("### Correlation Heatmap (Key Features)")
        corr_cols = [
            "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
            "temperature", "humidity", "wind_speed", "pressure", "visibility",
            "aqi_lag_1h", "aqi_lag_24h", "aqi_roll_mean_6h", "aqi_roll_max_24h",
            "aqi_change_1h", "aqi_change_24h",
        ]
        corr_cols = [c for c in corr_cols if c in df.columns]
        corr_df = df[corr_cols].corr()
        heat_fig = px.imshow(
            corr_df, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            template="plotly_dark",
        )
        heat_fig.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(heat_fig, width='stretch')

        # ── Row 4: Summary Stats Table ──────────────────────────────────────
        st.markdown("### 📋 Summary Statistics")
        summary_cols = [
            "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
            "temperature", "humidity", "wind_speed", "pressure",
        ]
        summary_cols = [c for c in summary_cols if c in df.columns]
        stats = df[summary_cols].describe().T
        stats["missing"] = df[summary_cols].isna().sum()
        stats["missing_pct"] = (stats["missing"] / len(df) * 100).round(1)
        st.dataframe(
            stats.style.format("{:.2f}").background_gradient(cmap="YlGnBu"),
            width='stretch',
        )

    except Exception as e:
        st.error(f"❌ EDA Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## 🏆 Model Performance Results")
    st.markdown("Results from the latest training run (v10).")

    # ── Performance Table ─────────────────────────────────────────────────
    perf_data = [
        # +24h
        {"Model": "Ridge", "Horizon": "+24h", "RMSE": 34.72, "MAE": 26.88, "R²": 0.0625, "MAPE": 22.43},
        {"Model": "RandomForest", "Horizon": "+24h", "RMSE": 35.98, "MAE": 28.14, "R²": -0.0066, "MAPE": 24.36},
        {"Model": "XGBoost", "Horizon": "+24h", "RMSE": 33.69, "MAE": 25.88, "R²": 0.1174, "MAPE": 23.02},
        {"Model": "LightGBM", "Horizon": "+24h", "RMSE": 34.02, "MAE": 26.53, "R²": 0.1001, "MAPE": 23.51},
        # +48h
        {"Model": "Ridge", "Horizon": "+48h", "RMSE": 38.03, "MAE": 30.13, "R²": -0.1263, "MAPE": 27.19},
        {"Model": "RandomForest", "Horizon": "+48h", "RMSE": 41.09, "MAE": 32.28, "R²": -0.3148, "MAPE": 30.65},
        {"Model": "XGBoost", "Horizon": "+48h", "RMSE": 37.32, "MAE": 26.54, "R²": -0.0843, "MAPE": 26.27},
        {"Model": "LightGBM", "Horizon": "+48h", "RMSE": 37.58, "MAE": 26.86, "R²": -0.0997, "MAPE": 26.68},
        # +72h
        {"Model": "Ridge", "Horizon": "+72h", "RMSE": 38.46, "MAE": 30.08, "R²": -0.1634, "MAPE": 27.81},
        {"Model": "RandomForest", "Horizon": "+72h", "RMSE": 44.88, "MAE": 33.84, "R²": -0.5846, "MAPE": 32.67},
        {"Model": "XGBoost", "Horizon": "+72h", "RMSE": 39.07, "MAE": 29.34, "R²": -0.2008, "MAPE": 27.93},
        {"Model": "LightGBM", "Horizon": "+72h", "RMSE": 38.06, "MAE": 28.81, "R²": -0.1393, "MAPE": 27.54},
        # Persistence baseline
        {"Model": "Persistence", "Horizon": "+1h", "RMSE": 10.19, "MAE": 2.08, "R²": 0.9193, "MAPE": 1.53},
    ]
    perf_df = pd.DataFrame(perf_data)

    st.markdown("### 📊 Performance Table")
    st.dataframe(
        perf_df.style.format({
            "RMSE": "{:.2f}", "MAE": "{:.2f}", "R²": "{:.4f}", "MAPE": "{:.2f}%",
        }).background_gradient(subset=["RMSE", "MAE", "MAPE"], cmap="YlOrRd")
        .background_gradient(subset=["R²"], cmap="RdYlGn"),
        width='stretch',
    )

    # ── Bar Chart Comparison (RMSE) ─────────────────────────────────────
    st.markdown("### RMSE Comparison by Model & Horizon")
    bar_df = perf_df[perf_df["Model"] != "Persistence"].copy()
    bar_chart = px.bar(
        bar_df, x="Model", y="RMSE", color="Horizon", barmode="group",
        template="plotly_dark",
        labels={"RMSE": "RMSE (lower is better)", "Model": "Model"},
    )
    bar_chart.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(bar_chart, width='stretch')

    # ── R² Comparison ───────────────────────────────────────────────────
    st.markdown("### R² Comparison by Model & Horizon")
    r2_chart = px.bar(
        bar_df, x="Model", y="R²", color="Horizon", barmode="group",
        template="plotly_dark",
        labels={"R²": "R² (higher is better)", "Model": "Model"},
    )
    r2_chart.add_hline(y=0, line_dash="dash", line_color="white", annotation_text="Baseline")
    r2_chart.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(r2_chart, width='stretch')

    st.markdown("---")
    st.markdown(
        "🏆 **Best model:** XGBoost at +24h horizon (RMSE = 33.69, R² = 0.117). "
        "Longer horizons show negative R², indicating AQI becomes harder to predict beyond 24 hours."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## 💡 SHAP Explainability")
    st.markdown("Feature importance for the +24h XGBoost model.")

    try:
        forecaster, df = load_model_and_data()
        df = df.tail(2000)

        drop_cols = [
            "timestamp", "weather_main",
            "target_aqi_24h", "target_aqi_48h", "target_aqi_72h",
        ]
        feature_cols = [c for c in df.columns if c not in drop_cols]

        # Use the +24h model
        model_24h = forecaster.models["24h"]

        # ── Model-native Feature Importance (safe, no SHAP segfault) ────
        st.markdown("### 🔥 Feature Importance (XGBoost built-in)")
        if hasattr(model_24h, "feature_importances_"):
            importance = model_24h.feature_importances_
            imp_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": importance,
            }).sort_values("Importance", ascending=True).tail(20)

            imp_bar = px.bar(
                imp_df, x="Importance", y="Feature", orientation="h",
                template="plotly_dark",
                color="Importance", color_continuous_scale="Blues",
            )
            imp_bar.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(imp_bar, width='stretch')

            st.markdown("### 📋 Top 10 Features")
            top10 = imp_df.sort_values("Importance", ascending=False).head(10)
            st.dataframe(
                top10.style.format({"Importance": "{:.4f}"})
                .background_gradient(cmap="Blues"),
                width='stretch',
            )
        else:
            st.info("Feature importance not available for this model type.")

        # ── Static SHAP images (generated during training) ─────────────
        st.markdown("### 🖼️ SHAP Summary Plots (from training run)")
        shap_summary_path = os.path.join(_repo_root, "shap_summary.png")
        shap_beeswarm_path = os.path.join(_repo_root, "shap_beeswarm.png")

        if os.path.exists(shap_summary_path):
            st.image(shap_summary_path, caption="SHAP Summary (Bar) — Mean absolute impact")
        else:
            st.info("SHAP summary image not found. Run `python pipelines/training_pipeline.py` to generate.")

        if os.path.exists(shap_beeswarm_path):
            st.image(shap_beeswarm_path, caption="SHAP Beeswarm — Distribution of impacts")
        else:
            st.info("SHAP beeswarm image not found. Run `python pipelines/training_pipeline.py` to generate.")

        # ── SHAP values JSON download ─────────────────────────────────
        shap_json_path = os.path.join(_repo_root, "shap_values.json")
        if os.path.exists(shap_json_path):
            st.markdown("### 📄 SHAP Values JSON")
            st.caption("Downloadable SHAP values for the last 200 test samples")
            with open(shap_json_path) as f:
                st.download_button(
                    label="Download SHAP Values",
                    data=f.read(),
                    file_name="shap_values.json",
                    mime="application/json",
                )

    except Exception as e:
        st.error(f"❌ SHAP Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🌫️ AQI Predictor for Karachi — Powered by XGBoost + Hopsworks Feature Store | "
    "Data: OpenWeatherMap, AQICN"
)
