import xarray as xr
import numpy as np
import json
from pathlib import Path
import re
import time


# === НАСТРОЙКИ ===
SWAN_DIR = Path(r"Z:\NOAA\predict_swan")
NOAA_DIR = Path(r"Z:\NOAA\predict_noaa")
OUT_JSON = Path(r"c:\swan_git\forecast_baltic.json")

TARGET_LAT = 55
TARGET_LON = 17

def find_latest_pred_file(folder: Path) -> Path:
    files = sorted(folder.glob("pred_NOAA_*.nc"))
    if not files:
        raise FileNotFoundError(f"Нет файлов pred_NOAA_*.nc в {folder}")
    return files[-1]


def parse_end_time_from_name(filename: str) -> np.datetime64:
    """
    pred_NOAA_2025-11-13_00h00m_2025-11-13_06h00m.nc
    → вернуть 2025-11-13T06:00:00
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


def main():

    while True:

        # === 1. Находим последний SWAN и соответствующий NOAA ===
        latest_swan = find_latest_pred_file(SWAN_DIR)
        latest_noaa = NOAA_DIR / latest_swan.name

        print("Используем:")
        print(" SWAN:", latest_swan)
        print(" NOAA:", latest_noaa)

        end_time = parse_end_time_from_name(latest_swan.name)
        print("Конец интервала:", end_time)

        # === 2. Загружаем ===
        ds_swan = xr.open_dataset(latest_swan)
        ds_noaa = xr.open_dataset(latest_noaa, decode_timedelta=False)

        # === 3. NOAA: исправляем время ===
        time_raw = ds_noaa["time"].values

        if np.issubdtype(time_raw.dtype, np.timedelta64):
            hours = time_raw / np.timedelta64(1, "h")
            print("NOAA: время как timedelta, преобразую")
        else:
            hours = time_raw.astype(float)
            print("NOAA: время числовое")

        # Первый шаг прогноза должен начинаться СРАЗУ после окончания интервала файла
        start_real_time = end_time
        print("Старт NOAA времени =", start_real_time)

        real_times = np.array(
            [start_real_time + np.timedelta64(int(h), "h") for h in hours],
            dtype="datetime64[m]"
        )

        ds_noaa = ds_noaa.assign_coords(time=("time", real_times))

        # === 4. Имена координат ===
        lat_swan, lon_swan = guess_lat_lon_names(ds_swan)
        lat_noaa, lon_noaa = guess_lat_lon_names(ds_noaa)

        # === 5. Обрезаем SWAN по времени ===
        ds_swan_sel = ds_swan.sel(time=slice(end_time, None))

        # NOAA уже нормализовано, оставляем всё
        ds_noaa_sel = ds_noaa

        # === 6. Общие временные шаги ===
        common_time = np.intersect1d(ds_swan_sel["time"].values, ds_noaa_sel["time"].values)
        print("Общие временные шаги:", common_time)

        ds_swan_sel = ds_swan_sel.sel(time=common_time)
        ds_noaa_sel = ds_noaa_sel.sel(time=common_time)

        # === 7. Берём одну точку ===
        point_swan = ds_swan_sel.sel({lat_swan: TARGET_LAT, lon_swan: TARGET_LON}, method="nearest")
        point_noaa = ds_noaa_sel.sel({lat_noaa: TARGET_LAT, lon_noaa: TARGET_LON}, method="nearest")

        # === 8. Переменные ===
        hs = point_swan["hs"]
        t2m = point_noaa["2m_temperature"]
        msl = point_noaa["mean_sea_level_pressure"]
        tp6 = point_noaa["total_precipitation_6hr"]
        sh  = point_noaa["specific_humidity"]
        u10 = point_noaa["10m_u_component_of_wind"]
        v10 = point_noaa["10m_v_component_of_wind"]

        # === 9. ПЕРЕВОДИМ В УДОБНЫЕ ЕДИНИЦЫ (С УДАЛЕНИЕМ ЛИШНИХ ОСЕЙ) ===

        times = point_noaa["time"].values
        time_str = [np.datetime_as_string(t, unit="m") for t in times]

        # волны — hs уже (time,) или (time,1?) — исправляем
        waves = np.round(point_swan["hs"].values.astype(float).squeeze(), 2).tolist()

        # температура K→°C
        temp_C = np.round((t2m.values.astype(float).squeeze() - 273.15), 1).tolist()

        # давление Pa→hPa
        pressure_hpa = np.round((msl.values.astype(float).squeeze() / 100.0), 1).tolist()

        # осадки m→mm
        rain_mm = np.round((tp6.values.astype(float).squeeze() * 1000.0), 2).tolist()

        # ветер в узлах
        wind_ms = np.sqrt(
            (u10.values.astype(float).squeeze()) ** 2 +
            (v10.values.astype(float).squeeze()) ** 2
        )
        wind_kt = np.round(wind_ms * 1.943844, 1).tolist()

        # облачность = максимум по уровням
        sh_vals = sh.values.astype(float)

        if sh_vals.ndim > 1:
            max_per_time = np.nanmax(sh_vals, axis=tuple(range(1, sh_vals.ndim)))
        else:
            max_per_time = sh_vals

        clouds = np.round(max_per_time, 0).astype(int).tolist()

        # === 10. CLOUDS = максимум по уровням (без нормализации!) ===
        sh_vals = sh.values.astype(float)
        print("shape specific humidity =", sh_vals.shape)

        # вариант 4D: (time, level, lat, lon)
        if sh_vals.ndim == 4:
            max_per_time = np.nanmax(sh_vals, axis=(1, 2, 3))

        # вариант 3D: (time, level, point) или (time, lat, lon)
        elif sh_vals.ndim == 3:
            max_per_time = np.nanmax(sh_vals, axis=(1, 2))

        # вариант 2D: (time, level)
        elif sh_vals.ndim == 2:
            max_per_time = np.nanmax(sh_vals, axis=1)

        # вариант 1D: (time)
        else:
            max_per_time = sh_vals

        # нормализация к 0–100%
        mn = float(np.nanmin(max_per_time))
        mx = float(np.nanmax(max_per_time))

        if mx > mn:
            clouds_pct = (max_per_time - mn) / (mx - mn) * 100.0
        else:
            clouds_pct = np.zeros_like(max_per_time)

        clouds = np.round(clouds_pct).astype(int).tolist()

        print("clouds final =", clouds)

        # === 11. Собираем JSON ===
        data_out = {
            "time": time_str,
            "waves": waves,
            "wind": wind_kt,
            "temp": temp_C,
            "rain": rain_mm,
            "clouds": clouds,
            "pressure": pressure_hpa
        }

        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(data_out, f, ensure_ascii=False, indent=2)

        print("===================================")
        print("  JSON записан в:", OUT_JSON)
        print("  Всего шагов:", len(time_str))
        print("===================================")

        time.sleep(600)


if __name__ == "__main__":
    main()
