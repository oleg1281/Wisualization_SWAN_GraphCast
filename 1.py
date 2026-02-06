import xarray as xr
import numpy as np
from pathlib import Path
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ======================================================
# НАСТРОЙКИ
# ======================================================
DATA_DIR = Path(r"c:\NOAA\SWAN_files")
LAT = 55
LON = 17.1
N_STEPS = 24
STEP_HOURS = 6

# ======================================================
# ЧТЕНИЕ ДАННЫХ
# ======================================================
out_m_fala, out_s_fala = [], []
out_m_wiatr, out_s_wiatr = [], []

files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".nc"))

for fname in files:
    with xr.open_dataset(DATA_DIR / fname) as ds:

        if "fala_stormgeo" not in ds:
            continue

        # --- волны ---
        era5_fala = ds["fala_era5"].sel(lat=LAT, lon=LON, method="nearest").values[:N_STEPS]
        mewo_fala = ds["fala_graphcast_swan"].sel(lat=LAT, lon=LON, method="nearest").values[:N_STEPS]
        stor_fala = ds["fala_stormgeo"].sel(lat=LAT, lon=LON, method="nearest").values[:N_STEPS]

        out_m_fala.append(np.abs(mewo_fala - era5_fala))
        out_s_fala.append(np.abs(stor_fala - era5_fala))

        # --- ветер ---
        era5_w = ds["WS_ERA5_ms"].sel(lat=LAT, lon=LON, method="nearest").values[:N_STEPS]
        mewo_w = ds["WS_Graphcast_ms"].sel(lat=LAT, lon=LON, method="nearest").values[:N_STEPS]
        stor_w = ds["WS_stormgeo_ms"].sel(lat=LAT, lon=LON, method="nearest").values[:N_STEPS]

        out_m_wiatr.append(np.abs(mewo_w - era5_w))
        out_s_wiatr.append(np.abs(stor_w - era5_w))

# ======================================================
# СРЕДНИЕ ЗНАЧЕНИЯ
# ======================================================
out_m_fala = np.array(out_m_fala)
out_s_fala = np.array(out_s_fala)
out_m_wiatr = np.array(out_m_wiatr)
out_s_wiatr = np.array(out_s_wiatr)

mean_m_fala = np.nanmean(out_m_fala, axis=0)
mean_s_fala = np.nanmean(out_s_fala, axis=0)
mean_m_wiatr = np.nanmean(out_m_wiatr, axis=0)
mean_s_wiatr = np.nanmean(out_s_wiatr, axis=0)

impr_fala = (1 - mean_m_fala / mean_s_fala) * 100
impr_wiatr = (1 - mean_m_wiatr / mean_s_wiatr) * 100

t = (np.arange(N_STEPS) + 1) * STEP_HOURS

# ======================================================
# FIGURE 1 — ВОЛНЫ
# ======================================================
fig_fala = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.65, 0.35],
    vertical_spacing=0.08,
    subplot_titles=[
        "MAE высоты волн (ERA5 — эталон)",
        "Относительное снижение ошибки MEWO (%)"
    ]
)

fig_fala.add_trace(
    go.Scatter(x=t, y=mean_m_fala, mode="lines+markers",
               name="MEWO", line=dict(width=3)),
    row=1, col=1
)

fig_fala.add_trace(
    go.Scatter(x=t, y=mean_s_fala, mode="lines+markers",
               name="StormGeo", line=dict(width=3)),
    row=1, col=1
)

fig_fala.add_trace(
    go.Scatter(x=t, y=impr_fala, mode="lines+markers",
               name="Преимущество MEWO",
               line=dict(width=3, color="#2ca02c")),
    row=2, col=1
)

fig_fala.add_hline(y=0, line_width=1, line_dash="dash", line_color="black", row=2)

fig_fala.update_layout(
    height=600,
    template="plotly_white",
    title="Сравнение прогноза высоты волн",
    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
)

fig_fala.update_yaxes(title_text="MAE (м)", row=1)
fig_fala.update_yaxes(title_text="Снижение MAE (%)", range=[-25, 25], row=2)
fig_fala.update_xaxes(title_text="Часы прогноза")

fig_fala.show()

# ======================================================
# FIGURE 2 — ВЕТЕР
# ======================================================
fig_w = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.65, 0.35],
    vertical_spacing=0.08,
    subplot_titles=[
        "MAE скорости ветра (ERA5 — эталон)",
        "Относительное снижение ошибки MEWO (%)"
    ]
)

fig_w.add_trace(
    go.Scatter(x=t, y=mean_m_wiatr, mode="lines+markers",
               name="MEWO", line=dict(width=3)),
    row=1, col=1
)

fig_w.add_trace(
    go.Scatter(x=t, y=mean_s_wiatr, mode="lines+markers",
               name="StormGeo", line=dict(width=3)),
    row=1, col=1
)

fig_w.add_trace(
    go.Scatter(x=t, y=impr_wiatr, mode="lines+markers",
               name="Преимущество MEWO",
               line=dict(width=3, color="#1f77b4")),
    row=2, col=1
)

fig_w.add_hline(y=0, line_width=1, line_dash="dash", line_color="black", row=2)

fig_w.update_layout(
    height=600,
    template="plotly_white",
    title="Сравнение прогноза скорости ветра",
    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
)

fig_w.update_yaxes(title_text="MAE (м/с)", row=1)
fig_w.update_yaxes(title_text="Снижение MAE (%)", range=[-25, 25], row=2)
fig_w.update_xaxes(title_text="Часы прогноза")

fig_w.show()
