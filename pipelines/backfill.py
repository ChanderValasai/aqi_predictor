
"""
Run once to populate the Feature Store with historical data.
AQICN provides historical data via their paid API;
OpenWeather Air Pollution API provides free history since Nov 27, 2020.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from feature_pipeline import compute_features, store_features

load_dotenv()


def fetch_openweather_history(lat, lon, start_dt, end_dt):
    """
    Fetch historical air pollution data from OpenWeatherMap.
    Free for any date since 2020-11-27.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    start_unix = int(start_dt.timestamp())
    end_unix   = int(end_dt.timestamp())
    url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={lat}&lon={lon}&start={start_unix}&end={end_unix}&appid={api_key}"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()["list"]


def backfill(days_back: int = 90):
    lat = float(os.getenv("CITY_LAT", 24.8607))
    lon = float(os.getenv("CITY_LON", 67.0011))

    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days_back)

    print(f"📅 Backfilling {days_back} days: {start_dt.date()} → {end_dt.date()}")
    records = fetch_openweather_history(lat, lon, start_dt, end_dt)
    print(f"   Retrieved {len(records)} hourly records")

    rows = []
    for r in records:
        ts = datetime.utcfromtimestamp(r["dt"])
        c  = r["components"]
        rows.append({
            "timestamp":   ts,
            "hour":        ts.hour,
            "day_of_week": ts.weekday(),
            "day_of_month":ts.day,
            "month":       ts.month,
            "is_weekend":  int(ts.weekday() >= 5),
            "aqi":         float(r["main"]["aqi"] * 50),  # convert 1-5 scale
            "pm25":        c.get("pm2_5", float("nan")),
            "pm10":        c.get("pm10",  float("nan")),
            "o3":          c.get("o3",    float("nan")),
            "no2":         c.get("no2",   float("nan")),
            "so2":         c.get("so2",   float("nan")),
            "co":          c.get("co",    float("nan")),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Compute lag/rolling features now that we have history
    from feature_pipeline import compute_lag_and_rolling_features
    df = compute_lag_and_rolling_features(df)
    df = df.dropna(subset=["target_aqi_24h"])  # drop rows with no future label

    store_features(df)
    print(f"✅ Backfill complete: {len(df)} training rows stored.")


if __name__ == "__main__":
    backfill(days_back=365)  # 1 year of history
