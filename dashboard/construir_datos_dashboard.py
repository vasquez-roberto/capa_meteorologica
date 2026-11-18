"""Construye dashboard/data.js a partir de los históricos del repositorio.

Lee los históricos interpolados por AGEB (PurpleAir y SIMA), las geometrías de AGEB de las capas
GeoJSON y los datos censales, y escribe un solo archivo compacto que index.html carga con
<script src="data.js">. Funciona abriendo index.html directamente, en GitHub Pages o en Vercel.

Uso (desde la raíz del repositorio o desde la carpeta dashboard):
    python dashboard/construir_datos_dashboard.py
Se puede agregar como último paso del workflow de GitHub Actions para que el dashboard se
actualice cada hora.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent


def encontrar_raiz(inicio=AQUI):
    for carpeta in [inicio, *inicio.parents]:
        if (carpeta / "SIMA").is_dir() and (carpeta / "Purple Air").is_dir():
            return carpeta
    raise FileNotFoundError("No se encontró la raíz del repositorio (carpetas 'SIMA' y 'Purple Air').")


RAIZ = encontrar_raiz()
SALIDA = AQUI / "data.js"
OFFSET_PURPLEAIR_H = -6          # PurpleAir se registra en UTC; SIMA ya está en hora local (UTC-6)
TOLERANCIA_GRADOS = 0.00012      # simplificación de polígonos (~13 m)
DECIMALES = 5
REPO_RAW = "https://raw.githubusercontent.com/vasquez-roberto/capa_meteorologica/main"
MUNICIPIOS = {
    "001": "Abasolo", "006": "Apodaca", "009": "Cadereyta Jiménez", "010": "El Carmen",
    "012": "Ciénega de Flores", "018": "García", "019": "San Pedro Garza García",
    "021": "General Escobedo", "025": "General Zuazua", "026": "Guadalupe", "031": "Juárez",
    "039": "Monterrey", "041": "Pesquería", "045": "Salinas Victoria",
    "046": "San Nicolás de los Garza", "047": "Hidalgo", "048": "Santa Catarina", "049": "Santiago",
}


# ---------------------------------------------------------------------------
# Geometría
# ---------------------------------------------------------------------------
def douglas_peucker(puntos, tol):
    puntos = np.asarray(puntos, dtype=float)
    if len(puntos) < 5:
        return puntos
    conservar = np.zeros(len(puntos), dtype=bool)
    conservar[[0, -1]] = True
    pila = [(0, len(puntos) - 1)]
    while pila:
        i, j = pila.pop()
        if j <= i + 1:
            continue
        a, b = puntos[i], puntos[j]
        seg = b - a
        largo = np.hypot(*seg)
        tramo = puntos[i + 1:j]
        if largo == 0:
            dist = np.hypot(*(tramo - a).T)
        else:
            dist = np.abs(seg[0] * (tramo[:, 1] - a[1]) - seg[1] * (tramo[:, 0] - a[0])) / largo
        k = int(np.argmax(dist))
        if dist[k] > tol:
            idx = i + 1 + k
            conservar[idx] = True
            pila += [(i, idx), (idx, j)]
    return puntos[conservar]


def simplificar_anillo(anillo):
    p = douglas_peucker(anillo, TOLERANCIA_GRADOS)
    if len(p) < 4:
        p = np.asarray(anillo, dtype=float)
    return np.round(p, DECIMALES).tolist()


def poligonos(geometria):
    """Lista de polígonos; cada polígono es una lista de anillos [[lon, lat], ...]."""
    if geometria is None:
        return []
    if geometria["type"] == "Polygon":
        partes = [geometria["coordinates"]]
    elif geometria["type"] == "MultiPolygon":
        partes = geometria["coordinates"]
    else:
        return []
    return [[simplificar_anillo(anillo) for anillo in parte] for parte in partes]


def leer_geojson(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)["features"]


# ---------------------------------------------------------------------------
# Series de tiempo
# ---------------------------------------------------------------------------
def a_hora_local(serie, offset_h):
    t = pd.to_datetime(serie, errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
    return t.dt.floor("h") + pd.Timedelta(hours=offset_h)


def matriz(df, columna, agebs, horas):
    """Matriz AGEB × hora con valores ×10 enteros (None = sin dato)."""
    p = (df.pivot_table(index="cvegeo", columns="timestamp", values=columna, aggfunc="last")
         .reindex(index=agebs, columns=horas))
    v = np.round(p.to_numpy(dtype=float) * 10)
    return [[None if np.isnan(x) else int(x) for x in fila] for fila in v]


def main():
    # --- Históricos ---
    pa = pd.read_csv(RAIZ / "Purple Air" / "interpolacion_calidadaire_historico.csv", encoding="utf-8-sig",
                     dtype={"cvegeo": str}, na_values=["ND"], low_memory=False,
                     usecols=["timestamp", "cvegeo", "pm25_interpolado", "pm10_interpolado"])
    pa["timestamp"] = a_hora_local(pa["timestamp"], OFFSET_PURPLEAIR_H)
    pa["cvegeo"] = pa["cvegeo"].str.strip().str.zfill(13)
    pa = pa.dropna(subset=["timestamp"]).drop_duplicates(["timestamp", "cvegeo"], keep="last")

    si = pd.read_csv(RAIZ / "SIMA" / "sima_interpolacion_historico.csv", encoding="utf-8-sig",
                     dtype={"CVEGEO": str}, na_values=["ND"], low_memory=False,
                     usecols=["timestamp", "CVEGEO", "PM25_ugm3", "PM10_ugm3"])
    si = si.rename(columns={"CVEGEO": "cvegeo"})
    si["timestamp"] = a_hora_local(si["timestamp"], 0)
    si["cvegeo"] = si["cvegeo"].str.strip().str.zfill(13)
    si = si.dropna(subset=["timestamp"]).drop_duplicates(["timestamp", "cvegeo"], keep="last")

    # --- Geometrías y censo ---
    geo_sima = {f["properties"]["CVEGEO"].zfill(13): f["geometry"]
                for f in leer_geojson(RAIZ / "SIMA" / "sima_PM25_AGEB.geojson")}
    feats_pa = leer_geojson(RAIZ / "Purple Air" / "AQ_PM25_v2.geojson")
    geo_pa, censo = {}, {}
    for f in feats_pa:
        p = f["properties"]
        clave = str(p.get("Clave Geográfica (CVEGEO)", "")).zfill(13)
        geo_pa[clave] = f["geometry"]
        censo[clave] = [int(p.get("Población Total") or 0), int(p.get("Niños de 0 a 5 años") or 0),
                        int(p.get("Mayores de 60 años") or 0), int(p.get("Personas con Discapacidad") or 0)]

    agebs = sorted((set(pa["cvegeo"]) | set(si["cvegeo"])) & (set(geo_sima) | set(geo_pa)))
    horas_pa = sorted(pa["timestamp"].unique())
    horas_si = sorted(si["timestamp"].unique())

    lista_agebs = []
    for clave in agebs:
        geom = geo_sima.get(clave) or geo_pa.get(clave)
        mun = clave[2:5]
        lista_agebs.append({
            "id": clave,
            "m": mun,
            "c": censo.get(clave, [0, 0, 0, 0]),
            "g": poligonos(geom),
        })

    # --- Sensores y estaciones (última lectura) ---
    sensores = [{"n": f["properties"].get("Sensor"), "lon": round(f["geometry"]["coordinates"][0], 5),
                 "lat": round(f["geometry"]["coordinates"][1], 5)}
                for f in leer_geojson(RAIZ / "Purple Air" / "purpleair_meteo.geojson")]
    estaciones = [{"n": f["properties"].get("Estacion") or f["properties"].get("Estación") or f["properties"].get("nombre"),
                   "lon": round(f["geometry"]["coordinates"][0], 5), "lat": round(f["geometry"]["coordinates"][1], 5)}
                  for f in leer_geojson(RAIZ / "SIMA" / "sima_meteo.geojson")]

    fmt = "%Y-%m-%dT%H:%M"
    datos = {
        "generado": pd.Timestamp.now(tz="America/Monterrey").strftime("%Y-%m-%d %H:%M"),
        "municipios": {k: v for k, v in MUNICIPIOS.items() if any(a["m"] == k for a in lista_agebs)},
        "agebs": lista_agebs,
        "pa": {"horas": [pd.Timestamp(h).strftime(fmt) for h in horas_pa],
               "pm25": matriz(pa, "pm25_interpolado", agebs, horas_pa),
               "pm10": matriz(pa, "pm10_interpolado", agebs, horas_pa)},
        "sima": {"horas": [pd.Timestamp(h).strftime(fmt) for h in horas_si],
                 "pm25": matriz(si, "PM25_ugm3", agebs, horas_si),
                 "pm10": matriz(si, "PM10_ugm3", agebs, horas_si)},
        "sensores": sensores,
        "estaciones": estaciones,
        "descargas": {
            "PurpleAir por AGEB (CSV)": f"{REPO_RAW}/Purple%20Air/interpolacion_calidadaire_historico.csv",
            "SIMA por AGEB (CSV)": f"{REPO_RAW}/SIMA/sima_interpolacion_historico.csv",
            "Lecturas de sensores PurpleAir (CSV)": f"{REPO_RAW}/Purple%20Air/purpleair_meteo_historico.csv",
            "Lecturas de estaciones SIMA (CSV)": f"{REPO_RAW}/SIMA/sima_meteo_historico.csv",
            "Capa PM2.5 PurpleAir por AGEB (GeoJSON)": f"{REPO_RAW}/Purple%20Air/AQ_PM25_v2.geojson",
            "Capa PM2.5 SIMA por AGEB (GeoJSON)": f"{REPO_RAW}/SIMA/sima_PM25_AGEB.geojson",
        },
        "repositorio": "https://github.com/vasquez-roberto/capa_meteorologica",
    }
    texto = json.dumps(datos, ensure_ascii=False, separators=(",", ":"))
    SALIDA.write_text("window.DASHBOARD_DATA=" + texto + ";\n", encoding="utf-8")
    print(f"{SALIDA.name}: {len(texto) / 1e6:.1f} MB | AGEBs {len(lista_agebs)} | "
          f"horas PurpleAir {len(horas_pa)} | horas SIMA {len(horas_si)} | "
          f"sensores {len(sensores)} | estaciones {len(estaciones)}")


if __name__ == "__main__":
    main()
