"""
Этот модуль используется FastAPI для получения прогноза.
Ничего не выполняется автоматически при импорте —
вся логика находится внутри функции load_forecast().
"""
import time

from pathlib import Path
import re
import numpy as np
import xarray as xr

# === НАСТРОЙКИ ===
SWAN_DIR = Path(r"Z:\NOAA\predict_swan")
NOAA_DIR = Path(r"Z:\NOAA\predict_noaa")


# ----------------------------------------------------------
# Функции-помощники (оставлены без изменений)
# ----------------------------------------------------------

def find_latest_pred_file(folder: Path) -> Path:
    """Возвращает последний файл pred_NOAA_*.nc"""
    files = sorted(folder.glob("pred_NOAA_*.nc"))
    if not files:
        raise FileNotFoundError(f"Нет файлов pred_NOAA_*.nc в {folder}")
    return files[-1]


def parse_end_time_from_name(filename: str) -> np.datetime64:
    """
    Парсит дату из имени файла:
    pred_NOAA_2025-11-13_00h00m_2025-11-13_06h00m.nc
                     конец интервала ↑
    """
    pattern = re.compile(
        r"pred_NOAA_"
        r"(\d{4}-\d{2}-\d{2})_(\d{2})h(\d{2})m_"
        r"(\d{4}-\d{2}-\d{2})_(\d{2})h(\d{2})m\.nc"
    )
    m = pattern.fullmatch(filename)
    if not m:
        raise ValueError(f"Имя не подходит: {filename}")

    end_date, end_H, end_M = m.group(4), m.group(5), m.group(6)
    return np.datetime64(f"{end_date}T{end_H}:{end_M}:00")


def guess_lat_lon_names(ds: xr.Dataset):
    """Находит названия координат LAT/LON в разных форматах."""
    for lat in ["lat", "latitude", "y"]:
        if lat in ds.coords:
            break
    else:
        raise KeyError("Нет координаты LAT")

    for lon in ["lon", "longitude", "x"]:
        if lon in ds.coords:
            break
    else:
        raise KeyError("Нет координаты LON")

    return lat, lon


# ----------------------------------------------------------
#  ГЛАВНАЯ ФУНКЦИЯ —
# ----------------------------------------------------------

def load_forecast(TARGET_LAT=54.3, TARGET_LON=18.6):
    """
    Основная функция — возвращает JSON прогноз.
    Здесь полностью используется ТВОЙ код, только обёрнут.
    """

    start = time.perf_counter()

    # === 1. Находим последний файл SWAN и соответствующий NOAA ===
    latest_swan = find_latest_pred_file(SWAN_DIR)
    latest_noaa = NOAA_DIR / latest_swan.name

    # конец предыдущего интервала
    end_time = parse_end_time_from_name(latest_swan.name)

    # === 2. Загружаем данные ===
    ds_swan = xr.open_dataset(latest_swan)
    ds_noaa = xr.open_dataset(latest_noaa, decode_timedelta=False)

    # === 3. NOAA — исправляем время ===
    time_raw = ds_noaa["time"].values

    if np.issubdtype(time_raw.dtype, np.timedelta64):
        hours = time_raw / np.timedelta64(1, "h")
    else:
        hours = time_raw.astype(float)

    start_real_time = end_time

    real_times = np.array(
        [start_real_time + np.timedelta64(int(h), "h") for h in hours],
        dtype="datetime64[m]"
    )

    ds_noaa = ds_noaa.assign_coords(time=("time", real_times))

    # === 4. Находим имена координат ===
    lat_swan, lon_swan = guess_lat_lon_names(ds_swan)
    lat_noaa, lon_noaa = guess_lat_lon_names(ds_noaa)

    # === 5. Обрезаем SWAN по времени ===
    ds_swan_sel = ds_swan.sel(time=slice(end_time, None))
    ds_noaa_sel = ds_noaa

    # === 6. Формируем общие временные шаги ===
    common_time = np.intersect1d(
        ds_swan_sel["time"].values,
        ds_noaa_sel["time"].values
    )
    #print("Общие временные шаги:", common_time)

    ds_swan_sel = ds_swan_sel.sel(time=common_time)
    ds_noaa_sel = ds_noaa_sel.sel(time=common_time)

    # === 7. Берём точку SWAN — только ближайшая, без интерполяции ===

    point_swan = ds_swan_sel.sel(
        {lat_swan: TARGET_LAT, lon_swan: TARGET_LON},
        method="nearest"
    )

    # Проверяем, есть ли там реальные данные
    hs_values = point_swan["hs"].values.astype(float)
    has_swan_data = not np.all(np.isnan(hs_values))

    # === NOAA — настоящая билинейная интерполяция ===
    point_noaa = ds_noaa_sel.sel(
        {lat_noaa: TARGET_LAT, lon_noaa: TARGET_LON},
        method="nearest"
    )

    # === 8. Переменные ===
    hs = point_swan["hs"]
    t2m = point_noaa["2m_temperature"]
    msl = point_noaa["mean_sea_level_pressure"]
    tp6 = point_noaa["total_precipitation_6hr"]
    sh = point_noaa["specific_humidity"]
    u10 = point_noaa["10m_u_component_of_wind"]
    v10 = point_noaa["10m_v_component_of_wind"]

    # === 9. Приводим всё в удобный формат ===
    times = point_noaa["time"].values
    time_str = [np.datetime_as_string(t, unit="m") for t in times]

    # === если SWAN не имеет данных, заполняем прочерками ===
    if not has_swan_data:
        waves = ["—"] * len(common_time)
    else:
        waves = np.round(hs_values.squeeze(), 2).tolist()
    temp_C = np.round((t2m.values.astype(float).squeeze() - 273.15), 1).tolist()
    pressure_hpa = np.round((msl.values.astype(float).squeeze() / 100.0), 1).tolist()
    rain_mm = np.round((tp6.values.astype(float).squeeze() * 1000.0), 2).tolist()

    # ветер
    wind_ms = np.sqrt(
        (u10.values.astype(float).squeeze()) ** 2 +
        (v10.values.astype(float).squeeze()) ** 2
    )
    wind_kt = np.round(wind_ms * 1.943844, 1).tolist()

    # облачность
    sh_vals = sh.values.astype(float)

    if sh_vals.ndim == 4:
        max_per_time = np.nanmax(sh_vals, axis=(1, 2, 3))
    elif sh_vals.ndim == 3:
        max_per_time = np.nanmax(sh_vals, axis=(1, 2))
    elif sh_vals.ndim == 2:
        max_per_time = np.nanmax(sh_vals, axis=1)
    else:
        max_per_time = sh_vals

    mn = float(np.nanmin(max_per_time))
    mx = float(np.nanmax(max_per_time))

    if mx > mn:
        clouds_pct = (max_per_time - mn) / (mx - mn) * 100.0
    else:
        clouds_pct = np.zeros_like(max_per_time)

    clouds = np.round(clouds_pct).astype(int).tolist()

    end = time.perf_counter()
    print(f"Время выполнения load_forecast: {end - start:.3f} сек")

    # === 10. Возвращаем JSON ===
    return {
        "time": time_str,
        "waves": waves,
        "wind": wind_kt,
        "temp": temp_C,
        "rain": rain_mm,
        "clouds": clouds,
        "pressure": pressure_hpa
    }


