from datetime import datetime
import json
import os
from pathlib import Path
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent


def cargar_env(ruta_env):
  if not ruta_env.exists():
    return
  for linea in ruta_env.read_text(encoding="utf-8-sig").splitlines():
    linea = linea.strip()
    if not linea or linea.startswith("#") or "=" not in linea:
      continue
    clave, valor = linea.split("=", 1)
    os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))


cargar_env(BASE_DIR / ".env")

API_KEY = os.getenv("API_KEY_PURPLEAIR")
CSV_SENSORES = BASE_DIR / "sensores_detectados.csv"
CAMPOS = "pm1.0,pm2.5,pm10.0,temperature,humidity"


def consultar_purpleair(sensor_index):
  if not API_KEY:
    raise RuntimeError("Falta API_KEY_PURPLEAIR en el archivo .env")
  url = f"https://api.purpleair.com/v1/sensors/{sensor_index}?fields={CAMPOS}"
  try:
    r = requests.get(url, headers={"X-API-Key": API_KEY}, timeout=15)
    r.raise_for_status()
    s = r.json().get("sensor", {})
    return (
        s.get("pm1.0"),
        s.get("pm2.5"),
        s.get("pm10.0"),
        s.get("temperature"),
        s.get("humidity"),
    )
  except Exception as e:
    print(f"Error consultando sensor {sensor_index}: {e}")
    return None, None, None, None, None


print("Consultando PurpleAir (Estandarizado a Red SIMA)...")
df_sensores = pd.read_csv(CSV_SENSORES).dropna(
    subset=["latitude", "longitude", "sensor_index"]
)

registros = []
features = []
timestamp_actual = datetime.now().strftime("%Y-%m-%d %H:00:00")

for _, row in df_sensores.iterrows():
  s_id = int(row["sensor_index"])
  nombre = row.get("name", f"Sensor {s_id}")
  pm1, pm25, pm10, temp_f, rh = consultar_purpleair(s_id)

  if pm25 is None:
    continue

  lat = float(row["latitude"])
  lon = float(row["longitude"])

  # Conversión estandarizada: Fahrenheit a Celsius
  temp_c = (
      round((float(temp_f) - 32) * 5 / 9, 2) if temp_f is not None else "ND"
  )

  registro = {
      "timestamp": timestamp_actual,
      "sensor_index": s_id,
      "nombre": str(nombre),
      "PM1_ugm3": float(pm1) if pm1 is not None else "ND",
      "PM25_ugm3": float(pm25) if pm25 is not None else "ND",
      "PM10_ugm3": float(pm10) if pm10 is not None else "ND",
      "Temperatura_C": temp_c,
      "Humedad_pct": float(rh) if rh is not None else "ND",
      "lat": lat,
      "lon": lon,
  }
  registros.append(registro)

  # GeoJSON estandarizado con la misma estructura visual que SIMA
  feature = {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [lon, lat]},
      "properties": {
          "Sensor": str(nombre),
          "ID Sensor": s_id,
          "PM1.0 (ug/m3)": registro["PM1_ugm3"],
          "PM2.5 (ug/m3)": registro["PM25_ugm3"],
          "PM10 (ug/m3)": registro["PM10_ugm3"],
          "Temperatura (C)": registro["Temperatura_C"],
          "Humedad Relativa (%)": registro["Humedad_pct"],
          "Fecha": timestamp_actual,
          "Red": "PurpleAir (Comunitaria)",
      },
  }
  features.append(feature)

df_purple = pd.DataFrame(registros)

# 1. CSV Actual estandarizado (sobrescribe)
CSV_ACTUAL = "purpleair_meteo_lecturas.csv"
df_purple.to_csv(CSV_ACTUAL, index=False, encoding="utf-8-sig")

# 2. CSV Histórico estandarizado (acumula)
CSV_HISTORICO = "purpleair_meteo_historico.csv"
if not os.path.exists(CSV_HISTORICO):
  df_purple.to_csv(CSV_HISTORICO, index=False, encoding="utf-8-sig")
else:
  df_purple.to_csv(
      CSV_HISTORICO, mode="a", header=False, index=False, encoding="utf-8-sig"
  )

# 3. GeoJSON estandarizado para ArcGIS
geojson_purple = {
    "type": "FeatureCollection",
    "crs": {
        "type": "name",
        "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
    },
    "features": features,
}

with open("purpleair_meteo.geojson", "w", encoding="utf-8") as f:
  json.dump(geojson_purple, f, ensure_ascii=False, indent=2)

print(
    f"Éxito PurpleAir: Datos convertidos a °C y estandarizados con SIMA.\nArchivos"
    f" generados: '{CSV_ACTUAL}', '{CSV_HISTORICO}' y 'purpleair_meteo.geojson'."
)