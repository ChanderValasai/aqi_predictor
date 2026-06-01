
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import hopsworks
import joblib
from datetime import datetime, timedelta

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

st.set_page_config(
    page_title="🌫️ AQI Predictor",
    page_icon="🌬️",
    layout="wide",
)

# ─── AQI COLOR SCALE ─────────────────────────────────────────────────────────
AQI_LEVELS = [
    (0,   50,  "Good",              "#00e400"),
    (51,  100, "Moderate",          "#ffff00"),
    (101, 150, "Unhealthy for SGs", "#ff7e00"),
    (151, 200, "Unhealthy",         "#ff0000"),
    (201, 300, "Very Unhealthy",    "#8f3f97"),
    (301, 500, "Hazardous",         "#7e0023"),
]

def get_aqi_category(aqi_val):
    for lo, hi, label, color in AQI_LEVELS:
        if lo <= aqi_val <= hi:
            return label, color
    return "Hazardous", "#7e0023"


@st.cache_resource(ttl=3600)
def load_model_and_data():
    """Load model from Hopsworks (cached for 1 hour)."""
    project = hopsworks.login(
        project=get_secret("HOPSWORKS_PROJECT"),
        api_key_value=get_secret("HOPSWORKS_API_KEY"),
    )
    mr = project.get_model_registry()
    model_meta = mr.get_model("aqi_forecaster", version=1)
    if model_meta is None:
        raise RuntimeError(
            "No model named **aqi_forecaster** found in your Hopsworks Model Registry. "
            "Please run the training pipeline first (`pipelines/training_pipeline.py`) "
            "to train and register a model."
        )
    model_dir  = model_meta.download()
    forecaster = joblib.load(f"{model_dir}/best_aqi_forecaster.pkl")

    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    df = df.sort_values("timestamp").tail(200)  # last 200 hours
    return forecaster, df


# ─── LAYOUT ───────────────────────────────────────────────────────────────────

st.title("🌫️ AQI Predictor Dashboard")
st.caption("3-Day Air Quality Forecast • Powered by ML")

try:
    forecaster, df = load_model_and_data()

    # ── Current AQI Card ──────────────────────────────────────────────────────
    current_aqi = int(df["aqi"].iloc[-1])
    label, color = get_aqi_category(current_aqi)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current AQI", current_aqi, label)
    with col2:
        st.metric("PM2.5", f'{df["pm25"].iloc[-1]:.1f} µg/m³')
    with col3:
        st.metric("PM10", f'{df["pm10"].iloc[-1]:.1f} µg/m³')
    with col4:
        st.metric("Temperature", f'{df["temperature"].iloc[-1]:.1f} °C')

    # Colored AQI status banner
    st.markdown(
        f'<div style="background:{color};padding:10px;border-radius:8px;'
        f'text-align:center;font-weight:bold;font-size:1.1em;">'
        f'Air Quality: {label}</div>',
        unsafe_allow_html=True
    )

    # ── Hazard Alert ─────────────────────────────────────────────────────────
    if current_aqi > 150:
        st.error(
            f"⚠️ **HAZARDOUS AQI ALERT**: Current level is {current_aqi}. "
            "Avoid outdoor activities. Wear N95 mask if going out."
        )

    st.divider()

    # ── 3-Day Forecast ────────────────────────────────────────────────────────
    drop_cols    = ["timestamp", "weather_main", "target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_df = df[feature_cols].copy()
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)
    X_latest     = X_df.iloc[[-1]].values

    predictions = forecaster.predict(X_latest)
    forecast_dates = [
        (datetime.now() + timedelta(hours=24)).strftime("%b %d"),
        (datetime.now() + timedelta(hours=48)).strftime("%b %d"),
        (datetime.now() + timedelta(hours=72)).strftime("%b %d"),
    ]

    st.subheader("📅 3-Day Forecast")
    fcol1, fcol2, fcol3 = st.columns(3)
    for i, (col, h, date) in enumerate(zip(
        [fcol1, fcol2, fcol3], [24, 48, 72], forecast_dates
    )):
        pred_val = int(np.asarray(predictions[h]).flat[0])
        lbl, clr = get_aqi_category(pred_val)
        with col:
            st.markdown(
                f'<div style="border:2px solid {clr};border-radius:8px;'
                f'padding:16px;text-align:center;">'
                f'<b>{date}</b><br>'
                f'<span style="font-size:2em;font-weight:bold;color:{clr};">'
                f'{pred_val}</span><br>{lbl}</div>',
                unsafe_allow_html=True
            )

    st.divider()

    # ── Historical AQI Chart ──────────────────────────────────────────────────
    st.subheader("📈 Historical AQI (Last 7 Days)")
    fig = px.line(
        df.tail(168), x="timestamp", y="aqi",
        title="AQI Over Time",
        labels={"aqi": "AQI", "timestamp": "Time"},
    )
    fig.add_hline(y=150, line_dash="dash", line_color="red",
                  annotation_text="Unhealthy Threshold")
    fig.add_hline(y=100, line_dash="dash", line_color="orange",
                  annotation_text="Moderate Threshold")
    st.plotly_chart(fig, use_container_width=True)

    # ── Pollutant Breakdown ───────────────────────────────────────────────────
    st.subheader("🧪 Pollutant Breakdown")
    pollutants = ["pm25", "pm10", "o3", "no2", "so2", "co"]
    latest_vals = [df[p].iloc[-1] for p in pollutants]
    bar_fig = px.bar(
        x=pollutants, y=latest_vals,
        labels={"x": "Pollutant", "y": "Concentration"},
        color=latest_vals,
        color_continuous_scale="RdYlGn_r",
        title="Current Pollutant Levels",
    )
    st.plotly_chart(bar_fig, use_container_width=True)

except Exception as e:
    project_set = bool(get_secret("HOPSWORKS_PROJECT"))
    api_key_set = bool(get_secret("HOPSWORKS_API_KEY"))
    if not project_set or not api_key_set:
        missing = []
        if not project_set:
            missing.append("`HOPSWORKS_PROJECT`")
        if not api_key_set:
            missing.append("`HOPSWORKS_API_KEY`")
        st.warning(
            f"⚙️ **Setup required** — the following secrets are not set: {', '.join(missing)}.\n\n"
            "**How to fix:**\n"
            "1. Open the 🔒 **Secrets** tab in the Replit sidebar\n"
            "2. Add `HOPSWORKS_PROJECT` (your Hopsworks project name)\n"
            "3. Add `HOPSWORKS_API_KEY` (from app.hopsworks.ai → avatar → Settings → API Keys)\n"
            "4. Restart the app"
        )
    else:
        st.error(f"❌ Error connecting to Hopsworks: {e}")
