from datetime import datetime
import json
import os
from pathlib import Path
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent

CSV_SENSORES = BASE_DIR / "sensores_detectados.csv"
CSV_ACTUAL = BASE_DIR / "purpleair_meteo_lecturas.csv"
CSV_HISTORICO = BASE_DIR / "purpleair_meteo_historico.csv"
GEOJSON_SALIDA = BASE_DIR / "purpleair_meteo.geojson"


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

API_KEY = os.getenv("API_KEY_PURPLEAIR") or os.getenv("PURPLEAIR_API_KEY")
CAMPOS = "pm1.0,pm2.5,pm10.0,temperature,humidity,name,latitude,longitude"


def consultar_purpleair_masivo(sensor_ids):
    """Consulta múltiples sensores en 1 sola llamada a la API ahorrando puntos."""
    if not API_KEY:
        raise RuntimeError("Falta API_KEY_PURPLEAIR en las variables de entorno")

    ids_str = ",".join(map(str, sensor_ids))
    url = f"https://api.purpleair.com/v1/sensors?show_only={ids_str}&fields={CAMPOS}"

    try:
        r = requests.get(url, headers={"X-API-Key": API_KEY}, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", [])
        fields = r.json().get("fields", [])

        # Convertir la matriz devuelta por la API masiva en DataFrame
        df_api = pd.DataFrame(data, columns=fields)
        return df_api
    except Exception as e:
        print(f"Error consultando lote de sensores PurpleAir: {e}")
        return pd.DataFrame()


print("Consultando PurpleAir en lote (Estandarizado a Red SIMA)...")

if not CSV_SENSORES.exists():
    raise FileNotFoundError(f"No se encontró el catálogo en {CSV_SENSORES}")

df_sensores = pd.read_csv(CSV_SENSORES).dropna(
    subset=["latitude", "longitude", "sensor_index"]
)

sensor_ids = df_sensores["sensor_index"].astype(int).unique().tolist()

# 1 SOLA PETICIÓN HTTP PARA TODOS LOS SENSORES
df_api = consultar_purpleair_masivo(sensor_ids)

registros = []
features = []
timestamp_actual = datetime.now().strftime("%Y-%m-%d %H:00:00")

if not df_api.empty:
    for _, row in df_api.iterrows():
        s_id = row.get("sensor_index")
        pm25 = row.get("pm2.5")

        if pd.isna(pm25) or pm25 is None:
            continue

        pm1 = row.get("pm1.0")
        pm10 = row.get("pm10.0")
        temp_f = row.get("temperature")
        rh = row.get("humidity")
        nombre = row.get("name", f"Sensor {s_id}")
        lat = float(row.get("latitude"))
        lon = float(row.get("longitude"))

        # Conversión de Fahrenheit a Celsius
        temp_c = (
            round((float(temp_f) - 32) * 5 / 9, 2)
            if temp_f is not None and not pd.isna(temp_f)
            else "ND"
        )

        registro = {
            "timestamp": timestamp_actual,
            "sensor_index": int(s_id),
            "nombre": str(nombre),
            "PM1_ugm3": float(pm1) if pd.notnull(pm1) else "ND",
            "PM25_ugm3": float(pm25) if pd.notnull(pm25) else "ND",
            "PM10_ugm3": float(pm10) if pd.notnull(pm10) else "ND",
            "Temperatura_C": temp_c,
            "Humedad_pct": float(rh) if pd.notnull(rh) else "ND",
            "lat": lat,
            "lon": lon,
        }
        registros.append(registro)

        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "Sensor": str(nombre),
                "ID Sensor": int(s_id),
                "PM1.0 (ug/m3)": registro["PM1_ugm3"],
                "PM2.5 (ug/m3)": registro["PM25_ugm3"],
                "PM10 (ug/m3)": registro["PM10_ugm3"],
                "Temperatura (C)": registro["Temperatura_C"],
                "Humedad Relativa (%)": registro["Humedad_pct"],
                "Fecha": timestamp_actual,
                "Red": "PurpleAir",
            },
        }
        features.append(feature)

df_purple = pd.DataFrame(registros)

# 1. CSV Actual estandarizado
df_purple.to_csv(CSV_ACTUAL, index=False, encoding="utf-8-sig")

# 2. CSV Histórico estandarizado (solo si hay registros)
if not df_purple.empty:
    if not CSV_HISTORICO.exists():
        df_purple.to_csv(CSV_HISTORICO, index=False, encoding="utf-8-sig")
    else:
        df_purple.to_csv(
            CSV_HISTORICO, mode="a", header=False, index=False, encoding="utf-8-sig"
        )

# 3. GeoJSON estandarizado
geojson_purple = {
    "type": "FeatureCollection",
    "crs": {
        "type": "name",
        "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
    },
    "features": features,
}

with open(GEOJSON_SALIDA, "w", encoding="utf-8") as f:
    json.dump(geojson_purple, f, ensure_ascii=False, indent=2)

print(
    f"Éxito PurpleAir: Datos convertidos a °C y estandarizados con SIMA.\nArchivos"
    f" generados en '{BASE_DIR.name}': '{CSV_ACTUAL.name}',"
    f" '{CSV_HISTORICO.name}' y '{GEOJSON_SALIDA.name}'."
)