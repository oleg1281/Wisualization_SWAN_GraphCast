import xarray as xr
import json
import numpy as np

# === Путь к твоему NetCDF-файлу (например, GraphCast) ===
path_nc = r"w:/Postprocesing/Oleh Bedenok/GRAPHCAST/GraphCast_hsig.nc"

# === Открываем файл ===
ds = xr.open_dataset(path_nc)

# === Берём только Балтийское море (примерный диапазон) ===
ds = ds.sel(latitude=slice(60, 54), longitude=slice(9, 31))


# === Средние значения по региону ===
def mean_safe(var):
    return float(ds[var].mean(skipna=True))


# === Собираем прогноз на шаги времени ===
times = ds.time.values.astype('datetime64[h]').astype(str)
forecast = {
    "time": [],
    "waves": [],
    "wind": [],
    "temp": [],
    "rain": [],
    "pressure": [],
}

for t in ds.time:
    subset = ds.sel(time=t)
    forecast["time"].append(str(t.values)[:16])
    forecast["waves"].append(round(float(subset["significant_height_of_combined_wind_waves_and_swell"].mean()), 2))

    # скорость ветра по компонентам
    u = subset["10m_u_component_of_wind"].mean()
    v = subset["10m_v_component_of_wind"].mean()
    forecast["wind"].append(round(float(np.sqrt(u ** 2 + v ** 2)), 1))

    forecast["temp"].append(round(float(subset["2m_temperature"].mean() - 273.15), 1))  # в °C
    forecast["rain"].append(round(float(subset["total_precipitation"].mean() * 1000), 1))  # мм
    forecast["pressure"].append(round(float(subset["mean_sea_level_pressure"].mean() / 100), 1))  # гПа

# === Сохраняем в JSON ===
with open("forecast_baltic.json", "w") as f:
    json.dump(forecast, f, indent=2)

print("✅ Сохранено: forecast_baltic.json")
