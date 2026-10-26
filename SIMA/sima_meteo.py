"""
Captura de lecturas horarias de la Red SIMA (Monterrey, N.L.)
PM10, PM2.5, Temperatura, Humedad Relativa y Radiación Solar.

Uso:
    python sima_captura.py            # captura normal
    python sima_captura.py --debug    # muestra endpoints descubiertos y JSON crudo
"""
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = Path(__file__).resolve().parent
CSV_ACTUAL = BASE_DIR / "sima_meteo_lecturas.csv"
CSV_HISTORICO = BASE_DIR / "sima_meteo_historico.csv"
GEOJSON_SALIDA = BASE_DIR / "sima_meteo.geojson"
CACHE_ENDPOINT = BASE_DIR / ".sima_endpoint_meteo.json"

DEBUG = "--debug" in sys.argv

BASE = "https://aire.nl.gob.mx/SIMA2017reportes/"
URL_REPORTE = BASE + "ReporteDiariosimaIcars_L3.php"
URL_CONC = BASE + "api_conc_correcto.php"

# Candidatos por si el descubrimiento automático no encuentra nada
CANDIDATOS_METEO = [
    "api_meteo_correcto.php",
    "api_meteo.php",
    "api_met_correcto.php",
    "api_meteorologia.php",
    "api_meteo_horario.php",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
    "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
}

# OJO: verifica en la web del SIMA el valor exacto de 'param' de San Pedro.
# "[SAN Pedro]" casi seguro es incorrecto; prueba "SANPEDRO" / "SAN PEDRO".
ESTACIONES = [
    ("CENTRO", "CENTRO", "Monterrey", 25.6760139, -100.338553, "CE"),
    ("SURESTE", "SURESTE", "Guadalupe", 25.6654972, -100.243653, "SE"),
    ("NORESTE", "NORESTE", "San Nicolás", 25.74503, -100.25317, "NE"),
    ("NOROESTE", "NOROESTE", "Monterrey", 25.7629306, -100.369578, "NO"),
    ("SUROESTE", "SUROESTE", "Santa Catarina", 25.6794444, -100.467831, "SO"),
    ("GARCIA", "NOROESTE 2", "García", 25.8004639, -100.585011, "NO2"),
    ("NORTE", "NORTE", "Escobedo", 25.7988361, -100.327164, "N"),
    ("NORESTE2", "NORESTE 2", "Apodaca", 25.777475, -100.1882, "NE2"),
    ("SURESTE2", "SURESTE 2", "Juárez", 25.646126, -100.095616, "SE2"),
    ("SANPEDRO", "SUROESTE 2", "San Pedro", 25.665275, -100.412853, "SO2"),
    ("SURESTE3", "SURESTE 3", "Cadereyta", 25.6008639, -99.9953028, "SE3"),
    ("NORTE2", "NORTE 2", "San Nicolás", 25.7297587, -100.310019, "N2"),
    ("SUR", "SUR", "Monterrey", 25.6169806, -100.273936, "S"),
    ("PESQUERIA", "ESTE", "Pesquería", 25.7905833, -100.078411, "NE3"),
    ("NOROESTE3", "NOROESTE 3", "García", 25.785, -100.46361112, "NO3"),
]


def a_num(val):
    try:
        if val is None or str(val).strip().upper() in ("", "ND", "N/D", "N/A", "NONE", "NULL"):
            return None
        return float(val)
    except (ValueError, TypeError):
        return None


def crear_sesion():
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1.5, status_forcelist=[500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update(HEADERS)
    return s


def _json_lista(resp):
    """Devuelve una lista de dicts si la respuesta es JSON válido, si no None."""
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    return data if isinstance(data, list) and data and isinstance(data[0], dict) else None


def clasificar_meteo(item):
    """Clasifica un registro como temp / rh / rs según su nombre."""
    p = str(item.get("Parameter", "")).strip().upper()
    largo = str(item.get("ParameterLargo", "")).strip().upper()
    texto = f"{p} {largo}"
    if p in ("TEMP", "TOUT", "T", "AMB_TEMP") or "TEMPERATURA" in texto:
        return "temp"
    if p in ("RH", "HR", "HUM") or "HUMEDAD" in texto:
        return "rh"
    if p in ("RS", "SR", "RAD", "SOLAR", "RSOL") or "RADIACI" in texto or "SOLAR" in texto:
        return "rs"
    return None


def descubrir_endpoint_meteo(session):
    """Busca el endpoint JSON de meteorología leyendo la página y sus scripts."""
    if CACHE_ENDPOINT.exists():
        try:
            url = json.loads(CACHE_ENDPOINT.read_text())["url"]
            r = session.get(url, params={"estacion1": "SURESTE"}, timeout=15)
            lista = _json_lista(r)
            if lista and any(clasificar_meteo(i) for i in lista):
                return url
        except Exception:
            pass

    candidatos = []
    try:
        html = session.get(URL_REPORTE, params={"estacion1": "SURESTE"}, timeout=15).text
        fuentes = [html]
        for src in re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', html, re.I):
            if "jquery" in src.lower() or "bootstrap" in src.lower():
                continue
            try:
                fuentes.append(session.get(urljoin(URL_REPORTE, src), timeout=15).text)
            except requests.RequestException:
                pass
        for txt in fuentes:
            for m in re.findall(r'["\']([^"\'\s]*\.php[^"\'\s]*)["\']', txt):
                ruta = m.split("?")[0]
                if "api" in ruta.lower() or "meteo" in ruta.lower() or "met" in ruta.lower():
                    candidatos.append(urljoin(URL_REPORTE, ruta))
    except requests.RequestException as e:
        print(f"Aviso: no se pudo leer la página del reporte: {e}")

    candidatos += [BASE + c for c in CANDIDATOS_METEO]
    vistos = []
    for c in candidatos:
        if c not in vistos and not c.endswith("api_conc_correcto.php"):
            vistos.append(c)

    if DEBUG:
        print("Endpoints candidatos:", *vistos, sep="\n  ")

    for url in vistos:
        try:
            r = session.get(url, params={"estacion1": "SURESTE"}, timeout=15)
        except requests.RequestException:
            continue
        lista = _json_lista(r)
        if lista and any(clasificar_meteo(i) for i in lista):
            CACHE_ENDPOINT.write_text(json.dumps({"url": url}))
            print(f"Endpoint meteorológico encontrado: {url}")
            return url
    return None


def leer_conc(session, param):
    pm10 = pm25 = fecha = None
    r = session.get(URL_CONC, params={"estacion1": param}, timeout=15)
    for item in _json_lista(r) or []:
        p = str(item.get("Parameter", "")).strip().upper()
        val = a_num(item.get("HrAveData"))
        if p.startswith("PM10"):
            pm10 = val
        elif p.startswith("PM2"):
            pm25 = val
        fecha = fecha or item.get("Date")
    return pm10, pm25, fecha


def leer_meteo(session, url, param):
    out = {"temp": None, "rh": None, "rs": None}
    fecha = None
    if not url:
        return out, fecha
    r = session.get(url, params={"estacion1": param}, timeout=15)
    lista = _json_lista(r) or []
    if DEBUG:
        print(f"[{param}] meteo crudo:", json.dumps(lista, ensure_ascii=False)[:500])
    for item in lista:
        clave = clasificar_meteo(item)
        if clave and out[clave] is None:
            val = item.get("HrAveData", item.get("Value", item.get("Valor")))
            out[clave] = a_num(val)
            fecha = fecha or item.get("Date")
    return out, fecha


def main():
    session = crear_sesion()
    url_meteo = descubrir_endpoint_meteo(session)
    if not url_meteo:
        print(
            "ADVERTENCIA: no se encontró el endpoint de meteorología.\n"
            "Abre la página del reporte en Chrome > F12 > Network > Fetch/XHR, "
            "recarga y copia la URL que devuelve temperatura; pégala en CANDIDATOS_METEO."
        )

    registros = []
    for param, zona, muni, lat, lon, clave in ESTACIONES:
        pm10 = pm25 = None
        met = {"temp": None, "rh": None, "rs": None}
        fecha = None
        try:
            pm10, pm25, fecha = leer_conc(session, param)
            met, f2 = leer_meteo(session, url_meteo, param)
            fecha = fecha or f2
        except Exception as e:
            print(f"Error en {zona}: {e}. Continuando...")

        registros.append({
            "timestamp": fecha or pd.Timestamp.now().strftime("%Y-%m-%d %H:00:00"),
            "Estacion": zona,
            "Clave": clave,
            "Municipio": muni,
            "PM10_ugm3": pm10,
            "PM25_ugm3": pm25,
            "Temperatura_C": met["temp"],
            "Humedad_pct": met["rh"],
            "RadiacionSolar_Wm2": met["rs"],
            "lat": lat,
            "lon": lon,
        })
        time.sleep(0.5)

    df = pd.DataFrame(registros)
    df.to_csv(CSV_ACTUAL, index=False, encoding="utf-8-sig", na_rep="ND")

    # Histórico sin duplicar (timestamp + estación)
    if CSV_HISTORICO.exists():
        hist = pd.read_csv(CSV_HISTORICO, encoding="utf-8-sig")
        hist = pd.concat([hist, df.fillna("ND")], ignore_index=True)
        hist = hist.drop_duplicates(subset=["timestamp", "Clave"], keep="last")
    else:
        hist = df.fillna("ND")
    hist.to_csv(CSV_HISTORICO, index=False, encoding="utf-8-sig")

    # GeoJSON (null en lugar de "ND" para que los SIG lo lean como numérico)
    def v(x):
        return None if pd.isna(x) else float(x)

    features = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
        "properties": {
            "Estacion": r.Estacion, "Clave": r.Clave, "Municipio": r.Municipio,
            "PM10 (ug/m3)": v(r.PM10_ugm3), "PM2.5 (ug/m3)": v(r.PM25_ugm3),
            "Temperatura (C)": v(r.Temperatura_C),
            "Humedad Relativa (%)": v(r.Humedad_pct),
            "Radiacion Solar (W/m2)": v(r.RadiacionSolar_Wm2),
            "Fecha": r.timestamp, "Red": "Red SIMA (Oficial)",
        },
    } for r in df.itertuples()]

    with open(GEOJSON_SALIDA, "w", encoding="utf-8") as f:
        json.dump({
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": features,
        }, f, ensure_ascii=False, indent=2)

    ok_met = df["Temperatura_C"].notna().sum()
    print(f"Listo: {len(df)} estaciones | con meteorología: {ok_met}")


if __name__ == "__main__":
    main()