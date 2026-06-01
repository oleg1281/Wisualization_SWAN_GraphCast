"""
Professional marine forecast PDF generator (3 models on each chart).
- Charts: temperature, wind, gusts, waves, clouds, pressure
- Each chart overlays up to 3 models: GraphCast, GFS, IFS
- Robust to missing data, wrong types ("-", None, strings), different lengths
- If all models are missing for a parameter (e.g., waves on land) -> chart is skipped (no blank space)
- Auto page breaks; re-draws header on new pages
- Returns PDF bytes (suitable for FastAPI response)
"""

from io import BytesIO
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FONTS_DIR = BASE_DIR / "fonts"

ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "mewo.png"

pdfmetrics.registerFont(
    TTFont("DejaVu", str(FONTS_DIR / "DejaVuSans.ttf"))
)

pdfmetrics.registerFont(
    TTFont("DejaVu-Bold", str(FONTS_DIR / "DejaVuSans-Bold.ttf"))
)

# -----------------------------
# Data cleaning helpers
# -----------------------------

def _parse_time_labels(time_list):
    """Convert forecast['time'] to compact labels; returns list[str]."""
    labels = []
    if not time_list:
        return labels

    for t in time_list:
        try:
            dt = datetime.fromisoformat(str(t).replace("Z", ""))
            labels.append(dt.strftime("%d.%m %Hh"))
        except Exception:
            # fallback: keep something readable
            s = str(t)
            labels.append(s[:16])
    return labels


def _clean_series(values, n):
    """
    Convert values (list-like) to list[float] length n, invalid -> np.nan.
    Returns None if:
      - values is empty/None
      - after cleaning all values are NaN
    Truncates or pads to length n.
    """
    if values is None:
        return None

    # allow numpy arrays, tuples etc.
    try:
        seq = list(values)
    except Exception:
        return None

    if len(seq) == 0:
        return None

    out = []
    for v in seq:
        try:
            # ignore obvious "no data" markers
            if v is None:
                out.append(np.nan)
                continue
            if isinstance(v, str) and v.strip() in ("-", "", "nan", "NaN", "None"):
                out.append(np.nan)
                continue

            fv = float(v)
            out.append(fv if not np.isnan(fv) else np.nan)
        except Exception:
            out.append(np.nan)

    # fit to n
    if len(out) >= n:
        out = out[:n]
    else:
        out = out + [np.nan] * (n - len(out))

    arr = np.array(out, dtype=float)
    if np.all(np.isnan(arr)):
        return None

    return out


# -----------------------------
# Plotting
# -----------------------------

def plot_multi_series_pro(
    title,
    x_labels,
    series_list,
    y_label,
    y_step=None,
):

    if not x_labels:
        return None

    n = len(x_labels)
    x = np.arange(n)

    fig = plt.figure(figsize=(8.27, 3.7), dpi=180)
    ax = fig.add_subplot(111)

    # ---------------- FORECAST RANGE ZONES ----------------
    # предполагаем шаг 6 часов (4 точки = 1 день)
    points_per_day = 4

    day3 = 3 * points_per_day
    day5 = 5 * points_per_day
    day7 = 7 * points_per_day

    # Зеленая зона (0–3 дня)
    ax.axvspan(0, min(day3, n), color="#00c853", alpha=0.07)

    # Желтая зона (3–5 дней)
    if n > day3:
        ax.axvspan(day3, min(day5, n), color="#ffd600", alpha=0.07)

    # Оранжевая зона (5–7 дней)
    if n > day5:
        ax.axvspan(day5, min(day7, n), color="#ff9100", alpha=0.07)

    # Красная зона (7+ дней)
    if n > day7:
        ax.axvspan(day7, n, color="#ff1744", alpha=0.05)
    # -------------------------------------------------------

    all_values = []
    plotted = 0

    for label, raw_values, color in series_list:
        y = _clean_series(raw_values, n)
        if y is None:
            continue

        y_arr = np.array(y, dtype=float)
        ax.plot(x, y_arr, color=color, linewidth=2.2, label=label)
        all_values.extend(y_arr[~np.isnan(y_arr)].tolist())
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel(y_label, fontsize=8)

    ax.tick_params(axis='y', labelsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(
        x_labels,
        rotation=90,
        ha="center",
        fontsize=5.4
    )
    ax.tick_params(axis='x', pad=0)

    # Контрольные уровни для быстрой визуальной оценки
    threshold = None
    threshold_label = None
    if "Wysokość fali" in title:
        threshold = 1.0
        threshold_label = "Poziom 1.0 m"
    elif "Wiatr (m/s)" in title:
        threshold = 10.0
        threshold_label = "Poziom 10 m/s"
    elif "Porywy" in title:
        threshold = 15.0
        threshold_label = "Poziom 15 m/s"
    elif "Temperatura" in title:
        threshold = 0.0
        threshold_label = "Poziom 0°C"

    if threshold is not None:
        ax.axhline(
            y=threshold,
            color="#d81b60",
            linestyle=(0, (6, 4)),
            linewidth=1.2,
            alpha=0.9,
            label=threshold_label
        )

    # ---------------- Y AXIS ----------------
    if y_step and all_values:

        if "Wiatr (m/s)" in title:
            ymin = 0
            ymax = 25

        elif "Porywy" in title:
            ymin = 0
            ymax = 35

        elif "Wysokość fali" in title:
            ymin = 0
            ymax = 4

        elif "Temperatura" in title:
            ymin = np.floor(min(all_values) / y_step) * y_step - 5
            ymax = np.ceil(max(all_values) / y_step) * y_step + 5

        else:
            ymin = np.floor(min(all_values) / y_step) * y_step
            ymax = np.ceil(max(all_values) / y_step) * y_step

        if np.isclose(ymin, ymax):
            ymin -= y_step
            ymax += y_step

        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(y_step))
    # -------------------------------------------------------

    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.grid(True, which="minor", linestyle="--", linewidth=0.4, alpha=0.18)
    ax.legend(fontsize=7, frameon=False)

    for spine in ax.spines.values():
        spine.set_alpha(0.35)

    # больше нижнего поля под вертикальные подписи дат
    fig.tight_layout(rect=(0, 0.13, 1, 1))

    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# -----------------------------
# PDF builder
# -----------------------------

def build_forecast_pdf(forecast: dict, lat: float, lon: float) -> bytes:
    """
    forecast dict expected keys:
      - time: list[str] (ISO) or list[anything convertible]
      - For each parameter key in ["temp","wind","gust","wave","cloud","pressure"]:
          <key>_gc, <key>_gfs, <key>_ifs -> list-like numeric or mixed
    Returns PDF bytes.
    """

    time_lbl = _parse_time_labels(forecast.get("time", []))

    # --- вычисляем время запуска модели ---
    time_list = forecast.get("time", [])

    if time_list:
        try:
            first_dt = datetime.fromisoformat(str(time_list[0]).replace("Z", ""))
            run_dt = first_dt - timedelta(hours=6)
            run_str = run_dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            run_str = "N/A"
    else:
        run_str = "N/A"

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    COLORS = {
        "GFS": "#2aa1ff",  # синий
        "IFS": "#111111",  # черный
        "IFS_SWAN": "#e5533d"  # красный
    }

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    def draw_header():
        # --- ЛОГОТИП ---
        if LOGO_PATH.exists():
            logo = ImageReader(str(LOGO_PATH))
            iw, ih = logo.getSize()

            target_h = 55
            target_w = target_h * iw / ih

            c.drawImage(
                logo,
                40,
                H - 90,
                target_w,
                target_h,
                mask='auto'
            )

        # --- RUN + КООРДИНАТЫ ---
        c.setFont("DejaVu-Bold", 11)  # было 9 → делаем 11 и жирный
        c.setFillColorRGB(0.1, 0.1, 0.1)  # почти чёрный

        c.drawString(
            40,
            H - 115,
            f"Run: {run_str}"
        )

        c.setFont("DejaVu", 10)

        c.drawString(
            40,
            H - 130,
            f"Lat: {lat:.4f}   Lon: {lon:.4f}"
        )

    def draw_footer(c, W):

        # --- ЛИЦЕНЗИЯ (маленький текст над линией) ---
        c.setFont("DejaVu", 5.5)
        c.setFillColorRGB(0.5, 0.5, 0.5)

        c.drawString(
            40,
            115,
            "Źródło: ECMWF IFS (Open Data, CC BY 4.0) oraz NOAA GFS (public domain). Dane przetworzone przez MEWO."
        )

        # --- ОРАНЖЕВАЯ ЛИНИЯ (фирменная) ---
        c.setStrokeColorRGB(0.91, 0.37, 0.17)  # фирменный оранжевый
        c.setLineWidth(1.2)
        c.line(40, 105, W - 40, 105)

        # --- СЕРАЯ ТОНКАЯ ЛИНИЯ НИЖЕ ---
        c.setStrokeColorRGB(0.8, 0.8, 0.8)
        c.setLineWidth(0.5)
        c.line(40, 100, W - 40, 100)

        # Основной текст светло-серый
        c.setFillColorRGB(0.45, 0.45, 0.45)

        # --- ЛЕВАЯ КОЛОНКА ---
        c.setFont("DejaVu-Bold", 9)
        c.drawString(40, 85, "MEWO S.A.")

        c.setFont("DejaVu", 6)

        left_block = [
            "Straszyn (83-010), ul. Starogardzka 17A,",
            "organ rejestrowy: Sąd Rejonowy Gdańsk-Północ w Gdańsku,",
            "numer KRS 0000625922,",
            "NIP 5833159342,",
            "kapitał zakładowy 200.000,00 zł opłacony w całości"
        ]

        y = 72
        for line in left_block:
            c.drawString(40, y, line)
            y -= 10

        # --- СРЕДНЯЯ КОЛОНКА ---
        middle_x = W / 2 - 60

        middle_block = [
            "biuro@mewo.eu",
            "www.mewo.eu",
        ]

        y = 85
        for line in middle_block:
            c.drawString(middle_x, y, line)
            y -= 10

        # Телефон выделен чуть темнее
        c.setFillColorRGB(0.35, 0.35, 0.35)
        c.drawString(middle_x, 65, "tel. +48 502 058 294")

        # --- НОМЕР СТРАНИЦЫ СПРАВА ---
        c.setFillColorRGB(0.45, 0.45, 0.45)
        c.setFont("DejaVu", 6)
        c.drawRightString(W - 40, 85, f"Strona {c.getPageNumber()}")

    def draw_summary():
        c.setFillColorRGB(0.94, 0.98, 1)
        c.roundRect(28, H - 215, W - 56, 60, 10, stroke=0, fill=1)

        c.setFillColorRGB(0.05, 0.12, 0.18)
        c.setFont("DejaVu-Bold", 11)
        c.drawString(40, H - 183, "Szybkie podsumowanie")

        c.setFont("DejaVu", 9)
        c.drawString(40, H - 201, "• Na wykresach porównanie: ECMWF IFS vs NOAA GFS")

    def draw_img(png, y_cursor):
        img = ImageReader(BytesIO(png))
        iw, ih = img.getSize()

        w = W - 56
        h = w * ih / iw

        c.drawImage(img, 28, y_cursor - h, w, h)
        return y_cursor - h - 18

    # first page
    draw_header()
    draw_footer(c, W)
    draw_summary()

    # if no time axis -> still valid PDF, but no charts
    if not time_lbl:
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.setFont("DejaVu", 12)
        c.drawString(28, H - 220, "Brak danych czasu (time) — nie mogę zbudować wykresów.")
        c.save()
        buf.seek(0)
        return buf.getvalue()

    # parameter specs: (title, unit, y_step, key_prefix)
    parameters = [
        ("Temperatura (°C)", "°C", 1, "temp"),
        ("Wiatr (m/s)", "m/s", 1, "wind"),
        ("Porywy wiatru (m/s)", "m/s", 1, "gust"),
        ("Wysokość fali (m)", "m", 0.2, "wave"),
        ("Zachmurzenie (%)", "%", 10, "cloud"),
        ("Ciśnienie (hPa)", "hPa", 5, "pressure"),
    ]

    y_cursor = H - 190

    charts_drawn = 0

    for title, unit, step, key in parameters:
        png = plot_multi_series_pro(
            title=title,
            x_labels=time_lbl,
            series_list=[
                ("ECMWF IFS", forecast.get(f"{key}_ifs"), COLORS["IFS"]),
                ("NOAA GFS",       forecast.get(f"{key}_gfs"), COLORS["GFS"]),
                ("IFS + SWAN", forecast.get(f"{key}_gc"), COLORS["IFS_SWAN"]),
            ],
            y_label=unit,
            y_step=step,
        )

        # no data for this parameter -> skip (no empty space)
        if not png:
            continue

        # page break if needed
        if y_cursor < 260:
            c.showPage()
            draw_header()
            draw_footer(c, W)
            y_cursor = H - 80

        y_cursor = draw_img(png, y_cursor)
        charts_drawn += 1

    # if absolutely no charts -> leave a note
    if charts_drawn == 0:
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.setFont("DejaVu", 12)
        c.drawString(28, H - 220, "Brak danych do wyświetlenia dla wybranego punktu.")

    c.save()
    buf.seek(0)
    return buf.getvalue()