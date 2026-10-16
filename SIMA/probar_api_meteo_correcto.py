import json
import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
}

session = requests.Session()
BASE_URL = "https://aire.nl.gob.mx/SIMA2017reportes/"

# 1. Probar api_meteo_correcto.php
print("--- 1. PROBANDO api_meteo_correcto.php (SURESTE) ---")
try:
  session.get(
      f"{BASE_URL}ReporteDiarioSimaM.php?datawr=SURESTE&estacion1=SURESTE",
      headers=HEADERS,
      timeout=10,
  )
  r = session.get(
      f"{BASE_URL}api_meteo_correcto.php?estacion1=SURESTE",
      headers=HEADERS,
      timeout=10,
  )
  print(json.dumps(r.json()[:5], indent=2, ensure_ascii=False))
except Exception as e:
  print(f"Error api_meteo_correcto.php: {e}")

# 2. Probar Meteorologia_actuales.json
print("\n--- 2. PROBANDO Meteorologia_actuales.json ---")
try:
  r = session.get(
      f"{BASE_URL}Meteorologia_actuales.json", headers=HEADERS, timeout=10
  )
  print(json.dumps(r.json()[:3], indent=2, ensure_ascii=False))
except Exception as e:
  print(f"Error Meteorologia_actuales.json: {e}")