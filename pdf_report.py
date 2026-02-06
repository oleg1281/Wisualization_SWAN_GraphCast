from io import BytesIO
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ---------- helpers ----------

def safe_float_array(arr):
    out = []
    if not arr:
        return out

    for v in arr:
        try:
            fv = float(v)
            if not np.isnan(fv):
                out.append(fv)
        except Exception:
            continue
    return out


def plot_single_series_pro(
    title, x_labels, y, y_label,
    color="#f5b000",
    y_step=None,
    fill=False,
    zones=None,
    y_tick_size=7,   # ← ДОБАВИТЬ
):

    if not y or len(y) != len(x_labels):
        return None

    # чуть выше график = красивее
    fig = plt.figure(figsize=(8.27, 3.7), dpi=180)
    ax = fig.add_subplot(111)

    x = np.arange(len(y))

    # фон (как на твоём втором графике)
    if zones:
        for (x0, x1, zcolor, alpha) in zones:
            ax.axvspan(x0, x1, color=zcolor, alpha=alpha, lw=0)

    ax.plot(x, y, color=color, linewidth=2.6)

    if fill:
        ax.fill_between(x, y, color=color, alpha=0.18)

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel(y_label, fontsize=7)
    ax.tick_params(axis="y", labelsize=y_tick_size)

    # Все даты, но компактнее: меньший шрифт + вертикально
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90, ha="center", fontsize=9)

    # Y шаг
    if y_step is not None:
        ymin = np.floor(min(y) / y_step) * y_step
        ymax = np.ceil(max(y) / y_step) * y_step
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(y_step))

    # Minor ticks + сетка как в “проф” графиках
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.grid(True, which="minor", linestyle="--", linewidth=0.4, alpha=0.18)

    # рамка спокойная
    for spine in ax.spines.values():
        spine.set_alpha(0.35)

    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------- main PDF ----------

def build_forecast_pdf(forecast: dict, lat: float, lon: float) -> bytes:
    time = forecast.get("time", [])
    time_lbl = []

    for t in time:
        try:
            dt = datetime.fromisoformat(t.replace("Z", ""))
            time_lbl.append(dt.strftime("%d.%m %H:%M"))
        except Exception:
            time_lbl.append(str(t)[:16])

    wind = safe_float_array(forecast.get("wind_ms"))
    waves = safe_float_array(forecast.get("waves"))
    temp = safe_float_array(forecast.get("temp"))
    rain = safe_float_array(forecast.get("rain"))
    clouds = safe_float_array(forecast.get("clouds"))
    pressure = safe_float_array(forecast.get("pressure"))

    def safe_minmax(arr):
        return (min(arr), max(arr)) if arr else (None, None)

    wmin, wmax = safe_minmax(wind)
    hmin, hmax = safe_minmax(waves)
    tmin, tmax = safe_minmax(temp)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    zones_wind = [
        (0, 8, "#00c853", 0.2),  # зелёная зона
        (8, 18, "#ffd600", 0.2),  # жёлтая зона
        (18, 28, "#0d47a1", 0.2),  # красная зона
        (28, 49, "#ff1744", 0.2),  # красная зона
    ]

    img_wind = plot_single_series_pro(
        "Wiatr (m/s)", time_lbl, wind, "m/s",
        color="#f5b000", y_step=1, fill=True,
        zones=zones_wind
    )

    img_wave = plot_single_series_pro(
        "Wysokość fali (m)", time_lbl, waves, "m",
        color="#2aa1ff", y_step=0.2, fill=True,
        zones=zones_wind
    )

    img_temp = plot_single_series_pro(
        "Temperatura (°C)", time_lbl, temp, "°C",
        color="#e5533d", y_step=1.0, fill=False,
        zones=zones_wind
    )

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    # header
    c.setFillColorRGB(0.02, 0.10, 0.16)
    c.rect(0, H - 90, W, 90, stroke=0, fill=1)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(28, H - 40, "Prognoza pogody — Morze Baltyckie")

    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.8, 0.95, 1)
    c.drawString(
        28, H - 62,
        f"Punkt: lat {lat:.4f}, lon {lon:.4f} | Wygenerowano: {now}"
    )

    # summary
    c.setFillColorRGB(0.94, 0.98, 1)
    c.roundRect(28, H - 160, W - 56, 60, 10, stroke=0, fill=1)

    c.setFillColorRGB(0.05, 0.12, 0.18)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, H - 128, "Szybkie podsumowanie")

    c.setFont("Helvetica", 10)
    y = H - 145
    for s in [
        f"Wiatr: {wmin:.1f}–{wmax:.1f} m/s" if wmin else None,
        f"Fale: {hmin:.1f}–{hmax:.1f} m" if hmin else None,
        f"Temperatura: {tmin:.1f}–{tmax:.1f} °C" if tmin else None,
    ]:
        if s:
            c.drawString(40, y, "• " + s)
            y -= 14

    def draw_img(png, y_cursor):
        if not png:
            return y_cursor
        img = ImageReader(BytesIO(png))
        iw, ih = img.getSize()
        w = W - 56
        h = w * ih / iw
        c.drawImage(img, 28, y_cursor - h, w, h)
        return y_cursor - h - 14

    # первые два графика — на первой странице
    y_cursor = H - 250
    y_cursor = draw_img(img_wind, y_cursor)
    y_cursor = draw_img(img_wave, y_cursor)

    # последний график — на новой странице
    c.showPage()
    y_cursor = H - 80
    y_cursor = draw_img(img_temp, y_cursor)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()
