from forecast import load_forecast
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
from pathlib import Path
import re
import numpy as np
import xarray as xr

app = FastAPI()

# --- Разрешаем запросы из браузера ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/api/forecast")
def get_forecast(lat: float=54.3, lon: float=18.6):
    lat, lon = round(lat, 1), round(lon, 1)
    forecast = load_forecast(TARGET_LAT=lat, TARGET_LON=lon)

    return forecast
