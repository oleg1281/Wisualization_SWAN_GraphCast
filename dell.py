import xarray as xr
from pathlib import Path
import os


dir = Path(r"\\smb-isilon.mewo.eu\mewo\Postprocesing\Oleh Bedenok\GRAPHCAST\NOAA\predict_noaa")


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
                #time.sleep(delay)
                pass
            else:
                break

    raise RuntimeError(f"Failed to open dataset {path}") from last_exc

TARGET_LAT = 54.3
TARGET_LON = 18.6

latest_noaa = dir/"pred_NOAA_2026-01-22_18h00m_2026-01-23_00h00m.nc"

with open_dataset_safe(latest_noaa, decode_timedelta=False) as ds_noaa:

    wind_ds = ds_noaa["10m_u_component_of_wind"]
