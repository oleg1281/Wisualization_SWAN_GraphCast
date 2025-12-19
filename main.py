from forecast import load_forecast
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
from pathlib import Path
import re
import numpy as np
import xarray as xr
from forecast import (
    load_forecast,
    load_temperature_grid,
    find_latest_pred_file,
    SWAN_DIR,
    NOAA_DIR
)

app = FastAPI()

# --- Разрешаем запросы из браузера ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

def normalize_lon_0360(lon):
    return lon % 360

@app.get("/api/forecast")
def get_forecast(lat: float=54.3, lon: float=18.6):
    # защита широты
    lat = max(-90.0, min(90.0, lat))
    lat = round(lat, 1)

    # нормализация долготы
    lon = normalize_lon_0360(lon)
    lon = round(lon, 1)
    forecast = load_forecast(TARGET_LAT=lat, TARGET_LON=lon)

    return forecast


@app.get("/api/temp_grid")
def get_temp_grid(time_idx: int = 0):
    latest_swan = find_latest_pred_file(SWAN_DIR)
    latest_noaa = NOAA_DIR / latest_swan.name

    ds_noaa = xr.open_dataset(latest_noaa, decode_timedelta=False)

    points = load_temperature_grid(ds_noaa)

    return {
        "points": points
    }