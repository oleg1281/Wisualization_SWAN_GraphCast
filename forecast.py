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
BASE_DIR = Path(r"/mnt/mewo/Postprocesing/Oleh Bedenok/GRAPHCAST/NOAA")

#SWAN_DIR = BASE_DIR / "predict_swan"
SWAN_DIR = BASE_DIR / "predict_swan_3h"
WAM_DIR = BASE_DIR / "predict_wam_fala"
#PREDICT_NOAA_DIR = BASE_DIR / "predict_noaa"
PREDICT_NOAA_DIR = BASE_DIR / "predict_ifs_3h"

GFS_DIR = BASE_DIR / "data_gfs_3h"
GFS_WAVE_DIR = BASE_DIR / "predict_gfs_fala_3h"

IFS_DIR = BASE_DIR / "predict_ifs_3h"
IFS_WAVE_DIR = BASE_DIR / "predict_ifs_3h"

AIFS_DIR = BASE_DIR / "predict_aifs"
AIFS_WAVE_DIR = BASE_DIR / "predict_ifs"  #----------------------------- нужно будет поменять когда добавлю сван к этой модели ----------------

#SWAN_DIR = Path("/mnt/mewo/Postprocesing/Oleh Bedenok/GRAPHCAST/NOAA/predict_swan")
#NOAA_DIR = Path("/mnt/mewo/Postprocesing/Oleh Bedenok/GRAPHCAST/NOAA/predict_noaa")

WAVE_HEIGHT_VARS = (
    "significant_wave_height",
    "significant_height_of_combined_wind_waves_and_swell",
    "swh",
)

TOTAL_SWELL_HEIGHT_VARS = (
    "significant_height_of_total_swell",
    "total_swell_significant_wave_height",
    "significant_height_of_swell_waves",
    "swell",
    "swell_height",
    "shts",
    "total_swell_height"
)

SWELL_COMPONENT_HEIGHT_VARS = (
    "significant_height_of_primary_swell",
    "significant_height_of_secondary_swell",
    "significant_height_of_tertiary_swell",
    "primary_swell_significant_wave_height",
    "secondary_swell_significant_wave_height",
    "tertiary_swell_significant_wave_height",
    "shps",
    "shss",
    "shts3",
    "swh1",
    "swh2",
    "swh3",
)


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


def find_latest_common_pred_file(model, run_offset: int = 0):
    """
    Возвращает последний общий pred_NOAA_*.nc с выбраной модели ветра и волны,
    """
    if model == "ifs":
        wiatr_fala = {f.name for f in IFS_DIR.glob("pred_NOAA_*.nc")}
        files = sorted(wiatr_fala)
        idx = len(files) - 1 - max(0, run_offset)
        if idx < 0:
            raise FileNotFoundError("❌ Недостаточно файлов прогноза для выбранного запуска")
        f = files[idx]
        wiatr = IFS_DIR / f
        fala = IFS_WAVE_DIR / f
        return wiatr, fala

    elif model == "gfs":
        wiatr = {f.name for f in GFS_DIR.glob("pred_NOAA_*.nc")}
        fala = {f.name for f in GFS_WAVE_DIR.glob("pred_NOAA_*.nc")}
        if not (fala & wiatr):
            raise FileNotFoundError("❌ Нет общего цикла во всех папках")
        files = sorted(fala & wiatr)
        idx = len(files) - 1 - max(0, run_offset)
        if idx < 0:
            raise FileNotFoundError("❌ Недостаточно файлов прогноза для выбранного запуска")
        f = files[idx]
        wiatr = GFS_DIR / f
        fala = GFS_WAVE_DIR / f
        return wiatr, fala

    elif model == "ifs_swan":
        fala = {f.name for f in SWAN_DIR.glob("pred_NOAA_*.nc")}
        wiatr = {f.name for f in PREDICT_NOAA_DIR.glob("pred_NOAA_*.nc")}
        if not (fala & wiatr):
            raise FileNotFoundError("❌ Нет общего цикла во всех папках")
        files = sorted(fala & wiatr)
        idx = len(files) - 1 - max(0, run_offset)
        if idx < 0:
            raise FileNotFoundError("❌ Недостаточно файлов прогноза для выбранного запуска")
        f = files[idx]
        wiatr = PREDICT_NOAA_DIR / f
        fala = SWAN_DIR / f
        return wiatr, fala

    elif model == "ifs_wam":
        fala = {f.name for f in WAM_DIR.glob("pred_NOAA_*.nc")}
        wiatr = {f.name for f in PREDICT_NOAA_DIR.glob("pred_NOAA_*.nc")}
        if not (fala & wiatr):
            raise FileNotFoundError("No common forecast files for wind and waves")
        files = sorted(fala & wiatr)
        idx = len(files) - 1 - max(0, run_offset)
        if idx < 0:
            raise FileNotFoundError("Requested forecast run offset is not available")
        f = files[idx]
        wiatr = PREDICT_NOAA_DIR / f
        fala = WAM_DIR / f
        return wiatr, fala

    return f"{model} Такая модель не найдена! "


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


def extract_swell_height(point_wave):
    total_var = next((name for name in TOTAL_SWELL_HEIGHT_VARS if name in point_wave.data_vars), None)
    if total_var is not None:
        return point_wave[total_var].values.astype(float)

    component_vars = [name for name in SWELL_COMPONENT_HEIGHT_VARS if name in point_wave.data_vars]
    if not component_vars:
        component_vars = [
            name for name in point_wave.data_vars
            if "swell" in name.lower()
            and any(token in name.lower() for token in ("height", "significant", "swh"))
            and not any(token in name.lower() for token in ("direction", "period", "mean", "peak"))
        ]
    component_arrays = [point_wave[name].values.astype(float) for name in component_vars]
    if not component_arrays:
        return None

    squared_sum = np.zeros_like(component_arrays[0], dtype=float)
    for arr in component_arrays:
        squared_sum = squared_sum + np.nan_to_num(arr, nan=0.0) ** 2
    return np.sqrt(squared_sum)


def deg_to_arrow(deg):
    arrows = ["↑", "↗", "→", "↘", "↓", "↙", "←", "↖"]
    return arrows[int((deg + 22.5) // 45) % 8]


# ----------------------------------------------------------
#  ГЛАВНАЯ ФУНКЦИЯ —
# ----------------------------------------------------------
def load_forecast(
    TARGET_LAT=54.2724,
    TARGET_LON=18.5861,
    model: str = "ifs_swan",
    run_offset: int = 0,
):
    """
    Основная функция — возвращает JSON прогноз.
    """
    # получение текущего времени для расчета скорости работы программы
    start = time.perf_counter()

    # === 1. Берём полный путь к ПОСЛЕДнему файл для этой конкретной модели===
    wiatr, fala = find_latest_common_pred_file(model, run_offset=run_offset)

    # === 2. Загружаем данные ===
    with open_dataset_safe(fala) as ds_fala, \
            open_dataset_safe(wiatr, decode_timedelta=False) as ds_wiatr:

        # === 6. Формируем общие временные шаги ===

        common_time = np.intersect1d(
            ds_fala["time"].values,
            ds_wiatr["time"].values
        )

        # обрезаем по общим временным шагам
        ds_fala = ds_fala.sel(time=common_time)
        ds_wiatr = ds_wiatr.sel(time=common_time)

        # возьмем только одну точку волны волн
        point_wave = ds_fala.sel(
            {"lat": TARGET_LAT, "lon": TARGET_LON},
            method="nearest"
        )

        wave_var = next((name for name in WAVE_HEIGHT_VARS if name in point_wave.data_vars), None)
        if wave_var is None:
            raise KeyError(
                f"No supported wave-height variable in {fala}. "
                f"Expected one of: {', '.join(WAVE_HEIGHT_VARS)}"
            )

        hs_values = point_wave[wave_var].values.astype(float)

        if np.isfinite(hs_values).any():
            waves = np.round(hs_values.squeeze(), 2).tolist()
        else:
            waves = None

        # возьмем только одну точку волны ветра
        swell_values = extract_swell_height(point_wave) if model == "ifs_wam" or model == "ifs_swan"  or model == "gfs"else None
        if swell_values is not None and np.isfinite(swell_values).any():
            swell_arr = np.round(swell_values.squeeze(), 2)
            swell = np.where(np.isfinite(swell_arr), swell_arr, None).tolist()
        else:
            swell = None

        point_wiatr = ds_wiatr.sel(
            {"lat": TARGET_LAT, "lon": TARGET_LON},
            method="nearest"
        )

        wind_ds = ds_wiatr[
            ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "total_precipitation_6hr"]
        ]

        wind_point = wind_ds.interp(
            {"lat": TARGET_LAT, "lon": TARGET_LON},
            method="linear"
        )

        # === 8. Переменные ===

        t2m = point_wiatr["2m_temperature"]
        msl = point_wiatr["mean_sea_level_pressure"]
        tp6 = wind_point["total_precipitation_6hr"]
        #sh = point_noaa["specific_humidity"]
        u10 = wind_point["10m_u_component_of_wind"]
        v10 = wind_point["10m_v_component_of_wind"]
        # --- ПОРЫВЫ ВЕТРА ---
        if "fg10" in point_wiatr.data_vars:
            fg10 = point_wiatr["fg10"]
        else:
            fg10 = None

        # --- ОБЛАЧНОСТЬ ---
        if "total_cloud_cover" in point_wiatr.data_vars:
            tcc = point_wiatr["total_cloud_cover"]
        else:
            tcc = None



        # === 9. Приводим всё в удобный формат ===
        times = point_wiatr["time"].values
        time_str = [np.datetime_as_string(t, unit="m") for t in times]

        t2m_vals = t2m.values.astype(float).squeeze()
        msl_vals = msl.values.astype(float).squeeze()
        temp_C = np.round((t2m_vals - 273.15), 1).tolist()
        pressure_hpa = np.round((msl_vals / 100.0), 1).tolist()
        scale = 5 if model == "gfs" else 1
        rain_mm = np.round((tp6.values.astype(float).squeeze() * 1000.0 * scale), 2).tolist() # zmenilem z 1000 na 3600




        # ветер
        u10_vals = u10.values.astype(float).squeeze()
        v10_vals = v10.values.astype(float).squeeze()
        wind_ms = np.sqrt(
            (u10_vals) ** 2 +
            (v10_vals) ** 2
        )

        wind_kt = np.round(wind_ms * 1.943844, 1).tolist()

        wind_ms = np.round(wind_ms, 1)

        # направление ветра (откуда дует)
        wind_dir = (
                           np.degrees(
                               np.arctan2(
                                   -u10_vals,
                                   -v10_vals
                               )
                           ) + 360
                   ) % 360
        wind_dir = np.round(wind_dir).astype(int)
        wind_dir_to = (wind_dir + 180) % 360
        wind_arrow = [deg_to_arrow(d) for d in wind_dir_to]

        # --- порывы ветра (gust) ---
        if fg10 is not None:

            arr = fg10.values.astype(float).squeeze()

            # если fg10 = NaN → берем wind_ms * 1.5
            gust_base = np.where(
                np.isnan(arr),
                wind_ms * 1.5,
                arr
            )

        else:
            # если fg10 вообще отсутствует
            gust_base = wind_ms * 1.5

        # итоговые массивы
        wind_porywy_ms = np.round(gust_base, 1).tolist()
        wind_porywy_kt = np.round(gust_base * 1.943844, 1).tolist()

        if tcc is None:
            total_cloud_cover = ["-"] * len(time_str)  # или len(common_time)
        else:
            total_cloud_cover = (tcc.values.squeeze() * 100).round().astype(int).tolist()

        end = time.perf_counter()
        print(f"Время выполнения load_forecast: {end - start:.3f} сек")

        # === 10. Возвращаем JSON ===
        if waves is None:
            return {
                "time": time_str,
                #"swell": swell,
                #"waves": waves,
                "wind_kt": wind_kt,
                "wind_ms": wind_ms.tolist(),
                "wind_porywy_kt": wind_porywy_kt,
                "wind_porywy_ms": wind_porywy_ms,
                "temp": temp_C,
                "rain": rain_mm,
                "pressure": pressure_hpa,
                "wind_arrow": wind_arrow,   # ←
                "total_cloud_cover": total_cloud_cover
            }
        else:
            return {
                "time": time_str,
                "waves": waves,
                "swell": swell,
                "wind_kt": wind_kt,
                "wind_ms": wind_ms.tolist(),
                "wind_porywy_kt": wind_porywy_kt,
                "wind_porywy_ms": wind_porywy_ms,
                "temp": temp_C,
                "rain": rain_mm,
                "pressure": pressure_hpa,
                "wind_arrow": wind_arrow,  # ←
                "total_cloud_cover": total_cloud_cover
            }

def load_forecast_all_models(TARGET_LAT=54.3, TARGET_LON=18.6):

    #print("Loading IFS_SWAN...")
    gc = load_forecast(TARGET_LAT, TARGET_LON, model="ifs_swan")

    print("Loading GFS...")
    gfs = load_forecast(TARGET_LAT, TARGET_LON, model="gfs")

    print("Loading IFS...")
    ifs = load_forecast(TARGET_LAT, TARGET_LON, model="ifs")

    return {
        "time": ifs.get("time"),

        # TEMPERATURE
        "temp_gc": gc.get("temp"),
        "temp_gfs": gfs.get("temp"),
        "temp_ifs": ifs.get("temp"),

        # WIND
        "wind_gc": gc.get("wind_ms"),
        "wind_gfs": gfs.get("wind_ms"),
        "wind_ifs": ifs.get("wind_ms"),

        # GUST
        "gust_gc": gc.get("wind_porywy_ms"),
        "gust_gfs": gfs.get("wind_porywy_ms"),
        "gust_ifs": ifs.get("wind_porywy_ms"),

        # WAVES
        "wave_gc": gc.get("waves"),
        "wave_gfs": gfs.get("waves"),
        "wave_ifs": ifs.get("waves"),

        # CLOUDS
        "cloud_gc": gc.get("total_cloud_cover"),
        "cloud_gfs": gfs.get("total_cloud_cover"),
        "cloud_ifs": ifs.get("total_cloud_cover"),

        # PRESSURE
        "pressure_gc": gc.get("pressure"),
        "pressure_gfs": gfs.get("pressure"),
        "pressure_ifs": ifs.get("pressure"),
    }
