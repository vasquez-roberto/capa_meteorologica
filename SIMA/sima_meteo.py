from datetime import datetime
import json
import os
import pandas as pd
import requests

ESTACIONES_SIMA = [
    {
        "param": "CENTRO",
        "zona": "CENTRO",
        "municipio": "Monterrey",
        "lat": 25.6760139,
        "lon": -100.338553,
        "clave": "CE",
    },
    {
        "param": "SURESTE",
        "zona": "SURESTE",
        "municipio": "Guadalupe",
        "lat": 25.6654972,
        "lon": -100.243653,
        "clave": "SE",
    },
    {
        "param": "NORESTE",
        "zona": "NORESTE",
        "municipio": "San Nicolás",
        "lat": 25.74503,
        "lon": -100.25317,
        "clave": "NE",
    },
    {
        "param": "NOROESTE",
        "zona": "NOROESTE",
        "municipio": "Monterrey",
        "lat": 25.7629306,
        "lon": -100.369578,
        "clave": "NO",
    },
    {
        "param": "SUROESTE",
        "zona": "SUROESTE",
        "municipio": "Santa Catarina",
        "lat": 25.6794444,
        "lon": -100.467831,
        "clave": "SO",
    },
    {
        "param": "GARCIA",
        "zona": "NOROESTE 2",
        "municipio": "García",
        "lat": 25.8004639,
        "lon": -100.585011,
        "clave": "NO2",
    },
    {
        "param": "NORTE",
        "zona": "NORTE",
        "municipio": "Escobedo",
        "lat": 25.7988361,
        "lon": -100.327164,
        "clave": "N",
    },
    {
        "param": "NORESTE2",
        "zona": "NORESTE 2",
        "municipio": "Apodaca",
        "lat": 25.777475,
        "lon": -100.1882,
        "clave": "NE2",
    },
    {
        "param": "SURESTE2",
        "zona": "SURESTE 2",
        "municipio": "Juárez",
        "lat": 25.646126,
        "lon": -100.095616,
        "clave": "SE2",
    },
    {
        "param": "[SAN Pedro]",
        "zona": "SUROESTE 2",
        "municipio": "San Pedro",
        "lat": 25.665275,
        "lon": -100.412853,
        "clave": "SO2",
    },
    {
        "param": "SURESTE3",
        "zona": "SURESTE 3",
        "municipio": "Cadereyta",
        "lat": 25.6008639,
        "lon": -99.9953028,
        "clave": "SE3",
    },
    {
        "param": "NORTE2",
        "zona": "NORTE 2",
        "municipio": "San Nicolás",
        "lat": 25.7297587,
        "lon": -100.310019,
        "clave": "N2",
    },
    {
        "param": "SUR",
        "zona": "SUR",
        "municipio": "Monterrey",
        "lat": 25.6169806,
        "lon": -100.273936,
        "clave": "S",
    },
    {
        "param": "PESQUERIA",
        "zona": "ESTE",
        "municipio": "Pesquería",
        "lat": 25.7905833,
        "lon": -100.078411,
        "clave": "NE3",
    },
    {
        "param": "NOROESTE3",
        "zona": "NOROESTE 3",
        "municipio": "García",
        "lat": 25.785,
        "lon": -100.46361112,
        "clave": "NO3",
    },
]

BASE_URL_REPORTE = (
    "https://aire.nl.gob.mx/SIMA2017reportes/ReporteDiariosimaIcars_L3.php"
)
BASE_URL_API_CONC = (
    "https://aire.nl.gob.mx/SIMA2017reportes/api_conc_correcto.php"
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
}


def a_num(val):
  try:
    return float(val) if val is not None else None
  except (ValueError, TypeError):
    return None


registros = []
timestamp_actual = datetime.now().strftime("%Y-%m-%d %H:00:00")
print("Consultando SIMA (Partículas + Meteorología)...")

for est in ESTACIONES_SIMA:
  session = requests.Session()
  url_estacion = f"{BASE_URL_REPORTE}?estacion1={est['param']}"
  session.get(url_estacion, headers=HEADERS, timeout=10)

  url_json = f"{BASE_URL_API_CONC}?estacion1={est['param']}"
  resp_json = session.get(url_json, headers=HEADERS, timeout=10)

  pm10 = pm25 = temp = rh = rs = None

  try:
    data = resp_json.json()
    for item in data:
      param = str(item.get("Parameter", "")).upper()
      val = item.get("HrAveData")

      if "PM10" in param:
        pm10 = a_num(val)
      elif "PM2" in param or "PM25" in param:
        pm25 = a_num(val)
      elif "TOUT" in param or "TMP" in param or "TEMP" in param:
        temp = a_num(val)
      elif "RH" in param or "HUM" in param:
        rh = a_num(val)
      elif "SRAD" in param or "RS" in param or "RAD" in param:
        rs = a_num(val)
  except Exception as e:
    print(f"Error en estación {est['zona']}: {e}")

  registros.append({
      "timestamp": timestamp_actual,
      "Estacion": est["zona"],
      "Clave": est["clave"],
      "Municipio": est["municipio"],
      "PM10_ugm3": pm10 if pm10 is not None else "ND",
      "PM25_ugm3": pm25 if pm25 is not None else "ND",
      "Temperatura_C": temp if temp is not None else "ND",
      "Humedad_pct": rh if rh is not None else "ND",
      "RadiacionSolar_Wm2": rs if rs is not None else "ND",
      "lat": est["lat"],
      "lon": est["lon"],
  })

df = pd.DataFrame(registros)

# 1. CSV Actual (sobrescribe)
CSV_ACTUAL = "sima_meteo_lecturas.csv"
df.to_csv(CSV_ACTUAL, index=False, encoding="utf-8-sig")

# 2. CSV Histórico (acumula)
CSV_HISTORICO = "sima_meteo_historico.csv"
if not os.path.exists(CSV_HISTORICO):
  df.to_csv(CSV_HISTORICO, index=False, encoding="utf-8-sig")
else:
  df.to_csv(
      CSV_HISTORICO, mode="a", header=False, index=False, encoding="utf-8-sig"
  )

# 3. GeoJSON para ArcGIS
features = []
for _, r in df.iterrows():
  feature = {
      "type": "Feature",
      "geometry": {
          "type": "Point",
          "coordinates": [float(r["lon"]), float(r["lat"])],
      },
      "properties": {
          "Estacion": str(r["Estacion"]),
          "Clave": str(r["Clave"]),
          "Municipio": str(r["Municipio"]),
          "PM10 (ug/m3)": r["PM10_ugm3"],
          "PM2.5 (ug/m3)": r["PM25_ugm3"],
          "Temperatura (C)": r["Temperatura_C"],
          "Humedad Relativa (%)": r["Humedad_pct"],
          "Radiacion Solar (W/m2)": r["RadiacionSolar_Wm2"],
          "Fecha": r["timestamp"],
          "Red": "Red SIMA (Oficial)",
      },
  }
  features.append(feature)

geojson_data = {
    "type": "FeatureCollection",
    "crs": {
        "type": "name",
        "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
    },
    "features": features,
}

with open("sima_meteo.geojson", "w", encoding="utf-8") as f:
  json.dump(geojson_data, f, ensure_ascii=False, indent=2)

print(
    f"Éxito SIMA: '{CSV_ACTUAL}', '{CSV_HISTORICO}' y 'sima_meteo.geojson'"
    " generados."
)