from datetime import datetime
import json
from pathlib import Path
import re
import pandas as pd
import requests

# Rutas absolutas dinámicas dentro de la subcarpeta 'SIMA'
BASE_DIR = Path(__file__).resolve().parent

CSV_ACTUAL = BASE_DIR / "sima_meteo_lecturas.csv"
CSV_HISTORICO = BASE_DIR / "sima_meteo_historico.csv"
GEOJSON_SALIDA = BASE_DIR / "sima_meteo.geojson"

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
BASE_URL_METEO = (
    "https://aire.nl.gob.mx/SIMA2017reportes/ReporteDiarioSimaM.php"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
}


def a_num(val):
  try:
    if val is None or str(val).strip().upper() in [
        "ND",
        "N/A",
        "N/D",
        "",
        "NONE",
    ]:
      return None
    return float(val)
  except (ValueError, TypeError):
    return None


def obtener_meteo_oficial_sima(session, param_est):
  """Captura los datos meteorológicos del informe diario oficial del SIMA."""
  temp = rh = rs = None
  try:
    # Paso 1: Petición inicial para cargar el contexto de la estación
    session.get(
        f"{BASE_URL_METEO}?estacion1={param_est}",
        headers=HEADERS,
        timeout=10,
    )

    # Paso 2: Petición con parámetro meteorológico para obtener el reporte completo
    url = f"{BASE_URL_METEO}?datawr={param_est}&estacion1={param_est}&parametro1=TEMP"
    resp = session.get(url, headers=HEADERS, timeout=12)

    # Extracción mediante expresiones regulares en la estructura HTML
    filas = re.findall(
        r"<tr[^>]*>(.*?)</tr>", resp.text, re.DOTALL | re.IGNORECASE
    )
    for fila in filas:
      celdas = re.findall(
          r"<t[dh][^>]*>(.*?)</t[dh]>", fila, re.DOTALL | re.IGNORECASE
      )
      celdas_limpias = [
          re.sub(r"<[^>]+>", "", c).strip() for c in celdas if c.strip()
      ]

      if len(celdas_limpias) >= 3:
        param_nombre = celdas_limpias[0].lower()
        val_horario = a_num(celdas_limpias[2])

        if "temperatura" in param_nombre and temp is None:
          temp = val_horario
        elif "humedad" in param_nombre and rh is None:
          rh = val_horario
        elif (
            "radiaci" in param_nombre or "solar" in param_nombre
        ) and rs is None:
          rs = val_horario
  except Exception as e:
    print(f"Error procesando meteorología oficial SIMA ({param_est}): {e}")

  return temp, rh, rs


registros = []
timestamp_actual = datetime.now().strftime("%Y-%m-%d %H:00:00")
print("Capturando datos de la Red SIMA (Partículas + Meteorología Oficial)...")

for est in ESTACIONES_SIMA:
  session = requests.Session()

  # 1. Extracción de Partículas Criterio
  url_estacion = f"{BASE_URL_REPORTE}?estacion1={est['param']}"
  session.get(url_estacion, headers=HEADERS, timeout=10)

  url_json = f"{BASE_URL_API_CONC}?estacion1={est['param']}"
  resp_json = session.get(url_json, headers=HEADERS, timeout=10)

  pm10 = pm25 = None
  try:
    data = resp_json.json()
    for item in data:
      param = str(item.get("Parameter", "")).strip().upper()
      val = item.get("HrAveData")
      if "PM10" in param:
        pm10 = a_num(val)
      elif "PM2" in param or "PM25" in param:
        pm25 = a_num(val)
  except Exception:
    pass

  # 2. Extracción de Meteorología Oficial SIMA
  temp, rh, rs = obtener_meteo_oficial_sima(session, est["param"])

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

# Exportaciones en la carpeta SIMA/
df.to_csv(CSV_ACTUAL, index=False, encoding="utf-8-sig")

if not CSV_HISTORICO.exists():
  df.to_csv(CSV_HISTORICO, index=False, encoding="utf-8-sig")
else:
  df.to_csv(
      CSV_HISTORICO, mode="a", header=False, index=False, encoding="utf-8-sig"
  )

features = []
for _, r in df.iterrows():
  features.append({
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
  })

geojson_data = {
    "type": "FeatureCollection",
    "crs": {
        "type": "name",
        "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
    },
    "features": features,
}

with open(GEOJSON_SALIDA, "w", encoding="utf-8") as f:
  json.dump(geojson_data, f, ensure_ascii=False, indent=2)

print(f"Éxito: Archivos SIMA generados en la carpeta '{BASE_DIR.name}'.")