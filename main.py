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
    find_latest_common_pred_file,
    SWAN_DIR,
    PREDICT_NOAA_DIR
)

from fastapi.responses import HTMLResponse
from fastapi import Request
from pathlib import Path
from fastapi.responses import JSONResponse

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


@app.get("/api/polygon_parts", response_class=JSONResponse)
def get_polygon_parts():
    static_dir = BASE_DIR / "static"
    parts = sorted(
        p.name
        for p in static_dir.glob("*.geojson")
    )
    return {"parts": parts}

@app.get("/api/forecast")

def get_forecast(
    lat: float = 54.3,
    lon: float = 18.6,
    model: str = "ifs_swan",   # ← ВАЖНО
    run_offset: int = 0
):
    lat = max(-90.0, min(90.0, lat))
    #lat = round(lat, 1)

    lon = normalize_lon_0360(lon)
    #lon = round(lon, 1)

    forecast = load_forecast(
        TARGET_LAT=lat,
        TARGET_LON=lon,
        model=model,            # ← ВАЖНО
        run_offset=max(0, run_offset)
    )
    return forecast


@app.get("/api/wind_grid")
def get_wind_grid(
    south: float,
    west: float,
    north: float,
    east: float,
    model: str = "ifs_swan",
    time_step: int = 0,
    run_offset: int = 0,
    spacing: float = 1.0,
):
    spacing = max(0.5, min(3.0, spacing))
    south, north = sorted((max(-90.0, south), min(90.0, north)))
    west, east = sorted((west, east))

    lat_values = np.arange(
        np.floor(south / spacing) * spacing,
        north + spacing,
        spacing,
    )
    lon_values = np.arange(
        np.floor(west / spacing) * spacing,
        east + spacing,
        spacing,
    )

    wiatr, _ = find_latest_common_pred_file(
        model,
        run_offset=max(0, run_offset),
    )

    with xr.open_dataset(wiatr, decode_timedelta=False, engine="netcdf4") as ds:
        time_index = max(0, min(time_step, ds.sizes["time"] - 1))
        coords = {
            "lat": xr.DataArray(lat_values, dims="grid_lat"),
            "lon": xr.DataArray(np.mod(lon_values, 360.0), dims="grid_lon"),
        }
        u_grid = ds["10m_u_component_of_wind"].isel(time=time_index).squeeze(drop=True).interp(coords)
        v_grid = ds["10m_v_component_of_wind"].isel(time=time_index).squeeze(drop=True).interp(coords)

        u_values = u_grid.values.astype(float)
        v_values = v_grid.values.astype(float)

    points = []
    for lat_index, lat in enumerate(lat_values):
        for lon_index, lon in enumerate(lon_values):
            u = u_values[lat_index, lon_index]
            v = v_values[lat_index, lon_index]
            if not np.isfinite(u) or not np.isfinite(v):
                continue
            points.append({
                "lat": round(float(lat), 3),
                "lon": round(float(lon), 3),
                "u": round(float(u), 2),
                "v": round(float(v), 2),
                "speed": round(float(np.hypot(u, v)), 2),
            })

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
