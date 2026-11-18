"""Construye dashboard/data.js a partir de los históricos del repositorio.

Lee los históricos interpolados por AGEB (PurpleAir y SIMA), las geometrías de AGEB de las capas
GeoJSON y los datos censales, y escribe un solo archivo compacto que index.html carga con
<script src="data.js">. Funciona abriendo index.html directamente, en GitHub Pages o en Vercel.

PurpleAir se publica siempre con la corrección EPA y el control de calidad:
- Si interpolacion_calidadaire_historico.csv ya tiene el formato nuevo (columna version_correccion),
  se usan sus valores tal cual.
- Si todavía tiene el formato anterior, este script aplica la misma corrección a las lecturas de
  cada sensor y vuelve a interpolar (mismas reglas que interpolacion_calidadaire_ageb.py).

Uso (desde la raíz del repositorio o desde la carpeta dashboard):
    python dashboard/construir_datos_dashboard.py
Se puede agregar como último paso del workflow de GitHub Actions para que el dashboard se
actualice cada hora.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as RutaPoligono
from scipy.interpolate import LinearNDInterpolator

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
# Corrección EPA y control de calidad de los sensores PurpleAir (versión "Corregida" del dashboard)
RANGO_PM25 = (0.0, 500.4)        # mismos límites que interpolacion_calidadaire_ageb.py
RANGO_PM10 = (0.0, 604.0)
RANGO_HR_SENSOR = (1, 99)        # 0 y 100 % del sensor suelen ser fallas; la corrección necesita humedad
VENTANA_QC = "7D"
QC = {"racha_congelada": 6, "fraccion_ceros": 0.5, "mediana_saturada": 300, "minimo_lecturas": 6,
      "factor_pico": 5, "minimo_pico": 50}
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


# ---------------------------------------------------------------------------
# Corrección EPA por sensor y nueva interpolación
# ---------------------------------------------------------------------------
def correccion_epa(pm, rh):
    """Corrección EPA por tramos (AirNow Fire and Smoke Map) con la humedad del propio sensor."""
    x = np.asarray(pm, dtype=float)
    rh = np.asarray(rh, dtype=float)
    w1 = x / 20 - 3 / 2
    w2 = x / 50 - 21 / 5
    tramos = [x < 30, (x >= 30) & (x < 50), (x >= 50) & (x < 210), (x >= 210) & (x < 260), x >= 260]
    valores = [
        0.524 * x - 0.0862 * rh + 5.75,
        (0.786 * w1 + 0.524 * (1 - w1)) * x - 0.0862 * rh + 5.75,
        0.786 * x - 0.0862 * rh + 5.75,
        (0.69 * w2 + 0.786 * (1 - w2)) * x - 0.0862 * rh * (1 - w2) + 2.966 * w2
        + 5.75 * (1 - w2) + 8.84e-4 * x ** 2 * w2,
        2.966 + 0.69 * x + 8.84e-4 * x ** 2,
    ]
    return np.clip(np.select(tramos, valores, default=np.nan), 0, None)


def primera_columna(df, candidatas):
    for c in candidatas:
        if c in df.columns:
            return c
    raise KeyError(f"Ninguna de las columnas {candidatas} está en el archivo")


def lecturas_sensores_corregidas():
    """Lecturas por sensor con control de calidad y PM2.5 corregido. PM10 solo lleva control de calidad
    (la EPA no publica corrección para PM10)."""
    df = pd.read_csv(RAIZ / "Purple Air" / "purpleair_meteo_historico.csv", encoding="utf-8-sig",
                     na_values=["ND"], low_memory=False)
    df = df.rename(columns={primera_columna(df, ["PM25_ugm3", "pm2_5_ugm3"]): "pm25",
                            primera_columna(df, ["PM10_ugm3", "pm10_0_ugm3"]): "pm10",
                            primera_columna(df, ["Humedad_pct", "humedad_pct"]): "hr"})
    df["timestamp"] = a_hora_local(df["timestamp"], OFFSET_PURPLEAIR_H)
    for c in ("pm25", "pm10", "hr", "lat", "lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (df.dropna(subset=["timestamp", "lat", "lon"])
          .drop_duplicates(["timestamp", "sensor_index"], keep="last")
          .sort_values(["sensor_index", "timestamp"]).reset_index(drop=True))
    total = len(df)
    df = df[df["pm25"].between(*RANGO_PM25) & df["pm10"].between(*RANGO_PM10)
            & df["hr"].between(*RANGO_HR_SENSOR)].reset_index(drop=True)

    # Reglas causales (solo la hora actual y las anteriores), iguales a las de producción
    excluir = pd.Series(False, index=df.index)
    catalogo = RAIZ / "Purple Air" / "sensores_detectados.csv"
    if catalogo.exists():
        tipo = pd.read_csv(catalogo).set_index("sensor_index")["location_type"]
        excluir |= df["sensor_index"].map(tipo).eq(1)                                   # interiores
    ventana = df.set_index("timestamp").groupby("sensor_index")["pm25"]
    conteo = ventana.rolling(VENTANA_QC).count().to_numpy()
    ceros = (df.assign(cero=df["pm25"].eq(0).astype(float)).set_index("timestamp")
             .groupby("sensor_index")["cero"].rolling(VENTANA_QC).mean().to_numpy())
    mediana = ventana.rolling(VENTANA_QC).median().to_numpy()
    suficiente = conteo >= QC["minimo_lecturas"]
    excluir |= suficiente & (ceros > QC["fraccion_ceros"])                              # sin señal
    excluir |= suficiente & (mediana > QC["mediana_saturada"])                          # saturado
    racha = df.groupby("sensor_index")["pm25"].diff().ne(0).cumsum()                 # congelado
    excluir |= (df.groupby(racha).cumcount() + 1) >= QC["racha_congelada"]
    mediana_hora = df.groupby("timestamp")["pm25"].transform("median")                 # picos aislados
    excluir |= (df["pm25"] > QC["factor_pico"] * mediana_hora) & (df["pm25"] > QC["minimo_pico"])

    df = df[~excluir].copy()
    df["pm25"] = correccion_epa(df["pm25"], df["hr"])
    print(f"Lecturas de sensores: {total:,} | válidas tras control de calidad: {len(df):,}")
    return df


def contiene(geometria, puntos):
    """Máscara de puntos (lon, lat) dentro de un Polygon o MultiPolygon, respetando huecos."""
    partes = [geometria["coordinates"]] if geometria["type"] == "Polygon" else geometria["coordinates"]
    dentro = np.zeros(len(puntos), dtype=bool)
    for anillos in partes:
        m = RutaPoligono(anillos[0]).contains_points(puntos)
        for hueco in anillos[1:]:
            m &= ~RutaPoligono(hueco).contains_points(puntos)
        dentro |= m
    return dentro


def centroide(geometria):
    """Centroide ponderado por área (equivalente al de shapely en coordenadas geográficas)."""
    partes = [geometria["coordinates"]] if geometria["type"] == "Polygon" else geometria["coordinates"]
    suma_a = suma_x = suma_y = 0.0
    for anillos in partes:
        for k, anillo in enumerate(anillos):
            p = np.asarray(anillo, dtype=float)
            x, y = p[:, 0], p[:, 1]
            cruz = x[:-1] * y[1:] - x[1:] * y[:-1]
            a = cruz.sum() / 2
            if a == 0:
                continue
            cx = ((x[:-1] + x[1:]) * cruz).sum() / (6 * a)
            cy = ((y[:-1] + y[1:]) * cruz).sum() / (6 * a)
            signo = 1 if k == 0 else -1
            a = abs(a) * signo
            suma_a += a; suma_x += cx * a; suma_y += cy * a
    return (suma_x / suma_a, suma_y / suma_a) if suma_a else (np.nan, np.nan)


def reinterpolar(lecturas, geometrias):
    """Réplica del método de producción con las lecturas corregidas: promedio de los sensores dentro
    del AGEB o interpolación lineal de Delaunay en su centroide. Devuelve horas y matrices AGEB × hora."""
    sitios = lecturas.drop_duplicates("sensor_index")[["sensor_index", "lon", "lat"]]
    xy = sitios[["lon", "lat"]].to_numpy()
    dentro = {}
    for i, g in enumerate(geometrias):
        m = contiene(g, xy)
        if m.any():
            dentro[i] = set(sitios["sensor_index"].to_numpy()[m])
    centros = np.array([centroide(g) for g in geometrias])
    horas = sorted(lecturas["timestamp"].unique())
    salida = {c: np.full((len(geometrias), len(horas)), np.nan) for c in ("pm25", "pm10")}
    for k, h in enumerate(horas):
        g = lecturas[lecturas["timestamp"] == h]
        if len(g) < 3:
            continue
        por_sitio = g.groupby(["lon", "lat"], as_index=False)[["pm25", "pm10"]].mean()
        valores = g.set_index("sensor_index")
        for c in ("pm25", "pm10"):
            try:
                lineal = LinearNDInterpolator(por_sitio[["lon", "lat"]].to_numpy(), por_sitio[c].to_numpy())(centros)
            except Exception:
                lineal = np.full(len(geometrias), np.nan)
            salida[c][:, k] = lineal
            for i, sensores in dentro.items():
                presentes = [s for s in sensores if s in valores.index]
                if presentes:
                    salida[c][i, k] = valores.loc[presentes, c].mean()
    return horas, salida


def a_lista(m):
    v = np.round(m * 10)
    return [[None if np.isnan(x) else int(x) for x in fila] for fila in v]


def main():
    # --- Históricos ---
    ruta_pa = RAIZ / "Purple Air" / "interpolacion_calidadaire_historico.csv"
    historico_corregido = "version_correccion" in pd.read_csv(ruta_pa, nrows=0, encoding="utf-8-sig").columns
    pa = pd.read_csv(ruta_pa, encoding="utf-8-sig", dtype={"cvegeo": str}, na_values=["ND"], low_memory=False,
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

    # --- PurpleAir: siempre con corrección EPA y control de calidad ---
    if historico_corregido:
        print("PurpleAir: se usan los valores del histórico (ya corregidos).")
        pa_pm25 = matriz(pa, "pm25_interpolado", agebs, horas_pa)
        pa_pm10 = matriz(pa, "pm10_interpolado", agebs, horas_pa)
    else:
        print("Aviso: el histórico de PurpleAir tiene el formato anterior. Se aplica aquí la corrección a cada "
              "sensor y se vuelve a interpolar. Corre reconstruir_historico_interpolacion.py para usar el histórico.")
        geometrias = [geo_sima.get(clave) or geo_pa.get(clave) for clave in agebs]
        horas_pa, epa = reinterpolar(lecturas_sensores_corregidas(), geometrias)
        pa_pm25, pa_pm10 = a_lista(epa["pm25"]), a_lista(epa["pm10"])

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
        "pa": {"horas": [pd.Timestamp(h).strftime(fmt) for h in horas_pa], "pm25": pa_pm25, "pm10": pa_pm10},
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
