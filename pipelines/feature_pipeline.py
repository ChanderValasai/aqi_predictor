
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv
import hopsworks

load_dotenv()

# ─── 1. FETCH RAW DATA ───────────────────────────────────────────────────────

def fetch_aqicn_data(city: str = "karachi") -> dict:
    """Fetch current AQI + pollutant data from AQICN."""
    token = os.getenv("AQICN_API_KEY")
    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "ok":
        raise ValueError(f"AQICN API error: {data}")
    return data["data"]


def fetch_weather_data(lat: float, lon: float) -> dict:
    """Fetch weather data from OpenWeatherMap."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ─── 2. COMPUTE FEATURES ─────────────────────────────────────────────────────

def compute_features(aqi_data: dict, weather_data: dict) -> pd.DataFrame:
    """
    Extract and engineer features from raw API responses.
    Returns a single-row DataFrame ready for the Feature Store.
    """
    now = datetime.now(timezone.utc)
    iaqi = aqi_data.get("iaqi", {})

    row = {
        # ── Timestamp ──
        "timestamp":       now,
        "hour":            now.hour,
        "day_of_week":     now.weekday(),        # 0=Monday … 6=Sunday
        "day_of_month":    now.day,
        "month":           now.month,
        "is_weekend":      int(now.weekday() >= 5),

        # ── Core AQI target ──
        "aqi":             float(aqi_data.get("aqi", np.nan)),

        # ── Pollutant sub-indices ──
        "pm25":            float(iaqi.get("pm25", {}).get("v", np.nan)),
        "pm10":            float(iaqi.get("pm10", {}).get("v", np.nan)),
        "o3":              float(iaqi.get("o3",   {}).get("v", np.nan)),
        "no2":             float(iaqi.get("no2",  {}).get("v", np.nan)),
        "so2":             float(iaqi.get("so2",  {}).get("v", np.nan)),
        "co":              float(iaqi.get("co",   {}).get("v", np.nan)),

        # ── Weather features ──
        "temperature":     weather_data["main"]["temp"],
        "humidity":        weather_data["main"]["humidity"],
        "wind_speed":      weather_data["wind"]["speed"],
        "wind_deg":        weather_data["wind"].get("deg", 0),
        "pressure":        weather_data["main"]["pressure"],
        "visibility":      weather_data.get("visibility", 10000) / 1000,  # km
        "cloud_cover":     weather_data["clouds"]["all"],
        "weather_main":    weather_data["weather"][0]["main"],
    }

    df = pd.DataFrame([row])

    # ── Cyclical encoding for time features ──
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ── Wind vector decomposition ──
    df["wind_u"] = df["wind_speed"] * np.cos(np.radians(df["wind_deg"]))
    df["wind_v"] = df["wind_speed"] * np.sin(np.radians(df["wind_deg"]))

    # ── One-hot encode weather condition ──
    weather_categories = ["Clear", "Clouds", "Rain", "Haze", "Fog", "Smoke", "Dust"]
    for cat in weather_categories:
        df[f"weather_{cat.lower()}"] = int(df["weather_main"].iloc[0] == cat)

    return df


def compute_lag_and_rolling_features(df_history: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features and rolling statistics.
    Requires a DataFrame sorted by timestamp with multiple rows.
    Call this AFTER accumulating history in the Feature Store.
    """
    df = df_history.sort_values("timestamp").copy()

    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"aqi_lag_{lag}h"]  = df["aqi"].shift(lag)
        df[f"pm25_lag_{lag}h"] = df["pm25"].shift(lag)

    for window in [3, 6, 12, 24]:
        df[f"aqi_roll_mean_{window}h"] = df["aqi"].rolling(window).mean()
        df[f"aqi_roll_std_{window}h"]  = df["aqi"].rolling(window).std()
        df[f"aqi_roll_max_{window}h"]  = df["aqi"].rolling(window).max()

    # AQI change rate
    df["aqi_change_1h"]  = df["aqi"].diff(1)
    df["aqi_change_3h"]  = df["aqi"].diff(3)
    df["aqi_change_24h"] = df["aqi"].diff(24)

    # Target: AQI 24h, 48h, 72h ahead (for 3-day forecast)
    df["target_aqi_24h"] = df["aqi"].shift(-24)
    df["target_aqi_48h"] = df["aqi"].shift(-48)
    df["target_aqi_72h"] = df["aqi"].shift(-72)

    return df


# ─── 3. STORE IN FEATURE STORE ───────────────────────────────────────────────

def store_features(df: pd.DataFrame):
    """Push engineered features to Hopsworks Feature Store."""
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    )
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        description="Hourly AQI and weather features",
        event_time="timestamp",
    )
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"✅ Stored {len(df)} row(s) to Feature Store at {datetime.now()}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    city  = os.getenv("CITY_NAME", "karachi")
    lat   = float(os.getenv("CITY_LAT", 24.8607))
    lon   = float(os.getenv("CITY_LON", 67.0011))

    print(f"📡 Fetching data for {city}...")
    aqi_data     = fetch_aqicn_data(city)
    weather_data = fetch_weather_data(lat, lon)

    df = compute_features(aqi_data, weather_data)
    print(f"📊 Computed {len(df.columns)} features")

    store_features(df)
