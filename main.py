from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from forecast import load_forecast_all_models

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

import logging
# --- Настройка логирования ---
logging.basicConfig(
    filename="access_log.txt",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)
import time

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


# add loging
@app.middleware("http")
async def log_requests(request: Request, call_next):

    start_time = time.perf_counter()

    # если nginx / proxy — берем реальный IP
    client_ip = request.headers.get("x-forwarded-for", request.client.host)
    user_agent = request.headers.get("user-agent", "unknown")
    path = request.url.path
    query = str(request.query_params)

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        logging.exception("❌ ERROR during request")
        raise e

    duration = time.perf_counter() - start_time

    log_line = (
        f"IP={client_ip} | "
        f"STATUS={status_code} | "
        f"PATH={path} | "
        f"QUERY={query} | "
        f"TIME={duration:.3f}s | "
        f"UA={user_agent}"
    )

    logging.info(log_line)

    return response


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
def forecast_pdf(lat: float = 54.2724, lon: float = 18.5861):

    lat = round(lat, 4)
    lon = round(lon, 4)

    forecast = load_forecast_all_models(
        TARGET_LAT=lat,
        TARGET_LON=lon,
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