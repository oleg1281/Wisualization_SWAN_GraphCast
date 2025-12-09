from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

# --- Разрешаем запросы из браузера ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/api/forecast")
def get_forecast():
    # 🟦 ВРЕМЕННО — просто читаем forecast_baltic.json
    with open("forecast_baltic.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
