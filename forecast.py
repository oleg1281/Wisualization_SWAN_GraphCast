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
PREDICT_NOAA_DIR = Path(r"Z:\NOAA\predict_noaa")
GFS_DIR = Path(r"Z:\NOAA\data_gfs")
GFS_WAVE_DIR = Path(r"Z:\NOAA\predict_gfs_fala")


# ----------------------------------------------------------
# Функции-помощники
# ----------------------------------------------------------

def open_dataset_safe(
    path,
    *,
    retries: int = 2,
    delay: float = 0.3,
    decode_timedelta: bool = False,
):
    last_exc = None

    for attempt in range(1, retries + 2):
        try:
            ds = xr.open_dataset(
                path,
                decode_timedelta=decode_timedelta,
                engine="netcdf4"
            )

            # 🔥 КРИТИЧЕСКАЯ ПРОВЕРКА
            if "time" not in ds.variables:
                ds.close()
                raise RuntimeError(
                    f"Dataset has no 'time'. Vars={list(ds.variables)}"
                )

            return ds

        except Exception as e:
            last_exc = e
            if attempt <= retries:
                time.sleep(delay)
            else:
                break

    raise RuntimeError(f"Failed to open dataset {path}") from last_exc


def get_noaa_dir(model: str) -> Path:
    if model == "gfs":
        return GFS_DIR
    return PREDICT_NOAA_DIR   # graphcast по умолчанию


def find_latest_common_pred_file():
    """
    Возвращает последний pred_NOAA_*.nc,
    который есть ВО ВСЕХ папках
    """

    swan_files = {f.name for f in SWAN_DIR.glob("pred_NOAA_*.nc")}
    noaa_files = {f.name for f in PREDICT_NOAA_DIR.glob("pred_NOAA_*.nc")}
    gfs_files  = {f.name for f in GFS_DIR.glob("pred_NOAA_*.nc")}
    gfsw_files = {f.name for f in GFS_WAVE_DIR.glob("pred_NOAA_*.nc")}

    common = swan_files & noaa_files & gfs_files & gfsw_files

    if not common:
        raise FileNotFoundError("❌ Нет общего цикла во всех папках")

    # сортировка по имени → самый новый в конце
    latest_name = sorted(common)[-1]

    return latest_name


def find_latest_gfs_wave_file(folder: Path) -> Path:
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


def load_temperature_grid(ds_noaa, time_idx=0):
    # 🔒 ЗАЩИТА: это должен быть NOAA-файл
    if "2m_temperature" not in ds_noaa.data_vars:
        raise ValueError(
            f"load_temperature_grid(): wrong dataset, variables = {list(ds_noaa.data_vars)}"
        )

    time_idx = int(np.clip(time_idx, 0, ds_noaa.sizes["time"] - 1))

    """
    Возвращает температуру 2m_temperature
    В УЗЛАХ СЕТКИ (1° × 1°), БЕЗ интерполяции
    """

    # координаты
    lat_name, lon_name = guess_lat_lon_names(ds_noaa)

    lats = ds_noaa[lat_name].values
    lons = ds_noaa[lon_name].values

    # температура в Кельвинах → °C
    t2m = ds_noaa["2m_temperature"].isel(time=time_idx) - 273.15
    t2m_vals = t2m.values.astype(float)

    points = []

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            if not (45 <= lat <= 60 and 0 <= lon <= 50):
                continue
            val = t2m_vals[time_idx, i, j]
            if np.isnan(val):
                continue

            points.append({
                "lat": float(lat),
                "lon": float(lon),
                "temp": round(float(val), 1)
            })

    return points


def deg_to_arrow(deg):
    arrows = ["↑", "↗", "→", "↘", "↓", "↙", "←", "↖"]
    return arrows[int((deg + 22.5) // 45) % 8]


# ----------------------------------------------------------
#  ГЛАВНАЯ ФУНКЦИЯ —
# ----------------------------------------------------------
def load_forecast(TARGET_LAT=54.3, TARGET_LON=18.6, model: str = "graphcast"):
    """
    Основная функция — возвращает JSON прогноз.
    Здесь полностью используется ТВОЙ код, только обёрнут.
    """
    start = time.perf_counter()

    # === 1. Берём ПОСЛЕДНИЙ ОБЩИЙ ЦИКЛ ===
    fname = find_latest_common_pred_file()

    latest_swan = SWAN_DIR / fname
    noaa_dir = get_noaa_dir(model)
    latest_noaa = noaa_dir / fname

    use_gfs_wave = (model == "gfs")
    latest_gfs_wave = find_latest_gfs_wave_file(GFS_WAVE_DIR) if use_gfs_wave else None

    # конец предыдущего интервала
    end_time = parse_end_time_from_name(latest_swan.name)

    # === 2. Загружаем данные ===
    with open_dataset_safe(latest_swan) as ds_swan, \
            open_dataset_safe(latest_noaa, decode_timedelta=False) as ds_noaa:

        # === FIX: приводим SWAN time к datetime64 ===
        if np.issubdtype(ds_swan["time"].dtype, np.timedelta64):
            ds_swan = ds_swan.assign_coords(
                time=end_time + ds_swan["time"]
            )

        #load_temperature_grid(ds_noaa)

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

        # === 🌊 ВОЛНЫ ===

        if use_gfs_wave:
            # === GFS WAVE ===
            latest_gfs_wave = find_latest_gfs_wave_file(GFS_WAVE_DIR)
            with xr.open_dataset(latest_gfs_wave) as ds_wave:

                lat_w, lon_w = guess_lat_lon_names(ds_wave)

                point_wave = ds_wave.sel(
                    {lat_w: TARGET_LAT, lon_w: TARGET_LON},
                    method="nearest"
                )

                hs_values = point_wave["hs"].values.astype(float)
                has_wave_data = not np.all(np.isnan(hs_values))

        else:
            # === SWAN ===
            point_swan = ds_swan_sel.sel(
                {lat_swan: TARGET_LAT, lon_swan: TARGET_LON},
                method="nearest"
            )

            hs_values = point_swan["hs"].values.astype(float)
            has_wave_data = not np.all(np.isnan(hs_values))

        # формируем waves один раз
        if not has_wave_data:
            waves = ["—"] * len(common_time)
        else:
            waves = np.round(hs_values.squeeze(), 2).tolist()

        # === NOAA — настоящая билинейная интерполяция ===

        point_noaa = ds_noaa_sel.sel(
            {lat_noaa: TARGET_LAT, lon_noaa: TARGET_LON},
            method="nearest"
        )

        wind_ds = ds_noaa_sel[
            ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "total_precipitation_6hr", ]
        ]

        wind_point = wind_ds.interp(
            {lat_noaa: TARGET_LAT, lon_noaa: TARGET_LON},
            method="linear"
        )

        # === 8. Переменные ===

        t2m = point_noaa["2m_temperature"]
        msl = point_noaa["mean_sea_level_pressure"]
        tp6 = wind_point["total_precipitation_6hr"]
        sh = point_noaa["specific_humidity"]
        u10 = wind_point["10m_u_component_of_wind"]
        v10 = wind_point["10m_v_component_of_wind"]

        # === 9. Приводим всё в удобный формат ===
        times = point_noaa["time"].values
        time_str = [np.datetime_as_string(t, unit="m") for t in times]

        temp_C = np.round((t2m.values.astype(float).squeeze() - 273.15), 1).tolist()
        pressure_hpa = np.round((msl.values.astype(float).squeeze() / 100.0), 1).tolist()
        scale = 10 if model == "gfs" else 1
        rain_mm = np.round((tp6.values.astype(float).squeeze() * 1000.0 * scale), 2).tolist()

        if model == "graphcast":
            rain_mm = [x if x >= 0.2 else 0 for x in rain_mm]
        elif model == "gfs":
            rain_mm = [x if x >= 0.15 else 0 for x in rain_mm]

        # ветер
        wind_ms = np.sqrt(
            (u10.values.astype(float).squeeze()) ** 2 +
            (v10.values.astype(float).squeeze()) ** 2
        )

        wind_kt = np.round(wind_ms * 1.943844).tolist()

        wind_ms = np.round(wind_ms, )

        # направление ветра (откуда дует)
        wind_dir = (
                           np.degrees(
                               np.arctan2(
                                   -u10.values.astype(float).squeeze(),
                                   -v10.values.astype(float).squeeze()
                               )
                           ) + 360
                   ) % 360
        wind_dir = np.round(wind_dir).astype(int)
        wind_dir_to = (wind_dir + 180) % 360
        wind_arrow = [deg_to_arrow(d) for d in wind_dir_to]

        end = time.perf_counter()
        print(f"Время выполнения load_forecast: {end - start:.3f} сек")

        # === 10. Возвращаем JSON ===
        return {
            "time": time_str,
            "waves": waves,
            "wind_kt": wind_kt,
            "wind_ms": wind_ms.tolist(),
            "temp": temp_C,
            "rain": rain_mm,
            "pressure": pressure_hpa,
            "wind_arrow": wind_arrow,   # ←
        }


