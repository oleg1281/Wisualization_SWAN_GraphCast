from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from io import BytesIO
from datetime import datetime

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pdf_report import build_forecast_pdf

#from reportlab.lib.pagesizes import A4
#from reportlab.pdfgen import canvas
#from reportlab.lib.utils import ImageReader
#from reportlab.lib import colors

from forecast import (
    load_forecast,
    load_temperature_grid,
    find_latest_common_pred_file,
    SWAN_DIR,
    PREDICT_NOAA_DIR
)

from fastapi.responses import HTMLResponse
from fastapi import Request
from pathlib import Path

from fastapi.staticfiles import StaticFiles


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

BASE_DIR = Path(__file__).parent
INDEX_HTML = BASE_DIR / "index.html"

# --- Разрешаем запросы из браузера ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

def normalize_lon_0360(lon):
    return lon % 360

@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML.read_text(encoding="utf-8")

@app.get("/api/forecast")

def get_forecast(
    lat: float = 54.3,
    lon: float = 18.6,
    model: str = "graphcast"   # ← ВАЖНО
):
    lat = max(-90.0, min(90.0, lat))
    lat = round(lat, 1)

    lon = normalize_lon_0360(lon)
    lon = round(lon, 1)

    forecast = load_forecast(
        TARGET_LAT=lat,
        TARGET_LON=lon,
        model=model            # ← ВАЖНО
    )
    return forecast


@app.get("/api/temp_grid")
def get_temp_grid(time_idx: int = 0):

    # 1️⃣ находим ПОСЛЕДНИЙ ОБЩИЙ ЦИКЛ
    fname = find_latest_common_pred_file()

    # 2️⃣ путь ТОЛЬКО к NOAA
    noaa_path = PREDICT_NOAA_DIR / fname

    # 3️⃣ открываем NetCDF ТОЛЬКО ЗДЕСЬ
    with xr.open_dataset(noaa_path, decode_timedelta=False) as ds_noaa:

        if "2m_temperature" not in ds_noaa.data_vars:
            raise ValueError(
                f"NOAA file has no 2m_temperature: {list(ds_noaa.data_vars)}"
            )

        points = load_temperature_grid(ds_noaa, time_idx=time_idx)

    # 4️⃣ ds_noaa здесь ГАРАНТИРОВАННО закрыт
    return {"points": points}


@app.get("/api/forecast_pdf")
def forecast_pdf(
    lat: float = 54.3,
    lon: float = 18.6,
    model: str = "graphcast"
):
    forecast = load_forecast(
        TARGET_LAT=round(lat, 4),
        TARGET_LON=round(lon, 4),
        model=model
    )

    pdf_bytes = build_forecast_pdf(forecast, lat, lon)
    filename = f"prognoza_{lat:.4f}_{lon:.4f}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )