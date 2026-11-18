"""Interpolación lineal de PM2.5 y PM10 por AGEB a partir de las lecturas de PurpleAir,
con control de calidad por sensor y corrección EPA del PM2.5 antes de interpolar.

Colocar en la subcarpeta 'Purple Air' y ejecutar después del script que actualiza
purpleair_meteo_lecturas.csv. No consulta la API ni modifica las lecturas.

Proceso por hora:
    1. Lecturas válidas (rango físico), igual que la versión anterior.
    2. Control de calidad por sensor, usando los últimos días del histórico de lecturas:
       sensores de interiores, sin señal (mayoría de ceros), saturados, congelados
       (mismo valor repetido) y picos aislados frente al resto de la red.
    3. Corrección EPA (EE. UU., Barkjohn et al. / AirNow Fire and Smoke Map) del PM2.5 de
       cada sensor con su propia humedad relativa. PM10 solo lleva control de calidad.
    4. Interpolación por AGEB: promedio de los sensores dentro del AGEB o, si no tiene,
       interpolación lineal de Delaunay en su centroide.
    También se interpola el valor sin corrección (método anterior) para comparar.

Salidas (en la subcarpeta 'Purple Air'):
    AQ_PM25_v2.geojson                         valor corregido + valor sin corrección
    AQ_PM10.geojson                            valor con control de calidad + sin corrección
    interpolacion_calidadaire_historico.csv    un registro por AGEB y lectura

Las funciones de este archivo también las usa reconstruir_historico_interpolacion.py, para
que el histórico reconstruido y el de tiempo real usen exactamente el mismo método.
"""
import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from shapely.geometry import mapping


# Definición de rutas adaptadas dinámicamente a la estructura de carpetas
BASE_DIR = Path(__file__).resolve().parent  # Subcarpeta 'Purple Air'
RAIZ_PROYECTO = BASE_DIR.parent  # Carpeta raíz del proyecto

# Archivos de entrada
CSV_PURPLEAIR = BASE_DIR / "purpleair_meteo_lecturas.csv"
CSV_SENSORES_HISTORICO = BASE_DIR / "purpleair_meteo_historico.csv"   # contexto del control de calidad
CSV_CATALOGO = BASE_DIR / "sensores_detectados.csv"                    # location_type: 1 = interiores
CSV_CENSO_INEGI = RAIZ_PROYECTO / "shp" / "cpv2020.csv"
ARCHIVO_SHP_COLONIAS = RAIZ_PROYECTO / "shp" / "2025_1_19_A.shp"

# Archivos de salida
SALIDA_GEOJSON_PM25 = BASE_DIR / "AQ_PM25_v2.geojson"
SALIDA_GEOJSON_PM10 = BASE_DIR / "AQ_PM10.geojson"
CSV_HISTORICO = BASE_DIR / "interpolacion_calidadaire_historico.csv"

MIN_PM = 0.0
MAX_PM25 = 500.4
MAX_PM10 = 604.0
MIN_SENSORES = 3

# Corrección y control de calidad
VERSION_CORRECCION = "EPA-US-2021+QC-v1"
RANGO_HR_SENSOR = (1, 99)      # 0 y 100 % del sensor suelen ser fallas
VENTANA_QC = "7D"              # días de historial para decidir si un sensor está sin señal o saturado
QC = {
    "racha_congelada": 6,      # a partir de la 6.ª lectura idéntica consecutiva, el sensor se considera congelado
    "fraccion_ceros": 0.5,     # sin señal si más de la mitad de sus lecturas en la ventana son 0
    "mediana_saturada": 300,   # saturado si su mediana en la ventana supera este valor (µg/m³)
    "minimo_lecturas": 6,      # lecturas mínimas en la ventana para aplicar las dos reglas anteriores
    "factor_pico": 5,          # pico si supera factor × mediana de la red en esa hora...
    "minimo_pico": 50,         # ...y además este valor absoluto (µg/m³)
}

COLUMNAS_LECTURAS = ["timestamp", "sensor_index", "PM25_ugm3", "PM10_ugm3", "Humedad_pct", "lat", "lon"]
# Nombres usados por versiones anteriores del histórico de lecturas
ALIAS_COLUMNAS = {"pm2_5_ugm3": "PM25_ugm3", "pm10_0_ugm3": "PM10_ugm3", "humedad_pct": "Humedad_pct"}
COLUMNAS_HISTORICO = [
    "timestamp", "cvegeo",
    "pm25_interpolado", "calidad_pm25",
    "pm10_interpolado", "calidad_pm10",
    "pm25_sin_correccion", "pm10_sin_correccion",
    "metodo", "n_sensores", "sensores_validos",
    "poblacion_total", "ninos_0a5", "adultos_mayores", "personas_discapacidad",
    "version_correccion",
]


# =============================================================================
# Clasificación
# =============================================================================
def clasificar_calidad_aire_pm25(valor):
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "Sin datos"
    if valor <= 15:
        return "Bueno"
    if valor <= 33:
        return "Aceptable"
    if valor <= 79:
        return "Mala"
    if valor <= 130:
        return "Muy alta"
    return "Extremadamente mala"


def clasificar_calidad_aire_pm10(valor):
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "Sin datos"
    if valor <= 45:
        return "Bueno"
    if valor <= 60:
        return "Aceptable"
    if valor <= 132:
        return "Mala"
    if valor <= 213:
        return "Muy alta"
    return "Extremadamente mala"


# =============================================================================
# Lecturas de sensores
# =============================================================================
def normalizar_lecturas(df):
    """Nombres de columna actuales, tipos numéricos y fecha como datetime (columna 't')."""
    df = df.rename(columns={k: v for k, v in ALIAS_COLUMNAS.items() if k in df.columns})
    faltantes = set(COLUMNAS_LECTURAS) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en las lecturas: {sorted(faltantes)}")
    df = df.copy()
    for columna in ("PM25_ugm3", "PM10_ugm3", "Humedad_pct", "lat", "lon"):
        df[columna] = pd.to_numeric(df[columna], errors="coerce")
    df["timestamp"] = df["timestamp"].astype(str).str.strip()
    df["t"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna(subset=["t", "lat", "lon"])


def filtrar_rango(df):
    """Mismo criterio que la versión anterior: PM2.5 y PM10 presentes y dentro del rango físico."""
    df = df.dropna(subset=["PM25_ugm3", "PM10_ugm3"])
    return df[df["PM25_ugm3"].between(MIN_PM, MAX_PM25) & df["PM10_ugm3"].between(MIN_PM, MAX_PM10)]


def leer_lecturas(ruta_csv):
    """Devuelve las lecturas válidas para interpolar. 'ND' y los valores fuera de
    rango se descartan; el CSV puede estar vacío si la última consulta falló."""
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró {ruta_csv}")
    try:
        df = pd.read_csv(ruta_csv, encoding="utf-8-sig", na_values=["ND"])
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=COLUMNAS_LECTURAS + ["t"])
    if df.empty:
        return pd.DataFrame(columns=COLUMNAS_LECTURAS + ["t"])
    total = len(df)
    df = filtrar_rango(normalizar_lecturas(df))
    if len(df) < total:
        print(f"Lecturas descartadas por dato faltante o fuera de rango: {total - len(df)}")
    return df.reset_index(drop=True)


def leer_historial_sensores(ruta_csv=CSV_SENSORES_HISTORICO):
    """Histórico de lecturas por sensor, con nombres de columna actuales y rango válido."""
    if not ruta_csv.exists():
        return pd.DataFrame(columns=COLUMNAS_LECTURAS + ["t"])
    df = pd.read_csv(ruta_csv, encoding="utf-8-sig", na_values=["ND"], low_memory=False)
    df = filtrar_rango(normalizar_lecturas(df))
    return df.drop_duplicates(["timestamp", "sensor_index"], keep="last").reset_index(drop=True)


# =============================================================================
# Control de calidad y corrección EPA
# =============================================================================
def correccion_epa(pm25, humedad):
    """Corrección EPA de EE. UU. por tramos (AirNow Fire and Smoke Map).

    0–30:    0.524x − 0.0862·HR + 5.75
    30–50:   transición lineal de la pendiente 0.524 a 0.786
    50–210:  0.786x − 0.0862·HR + 5.75
    210–260: transición hacia el tramo de humo intenso
    ≥ 260:   2.966 + 0.69x + 8.84e-4·x²
    """
    x = np.asarray(pm25, dtype=float)
    rh = np.asarray(humedad, dtype=float)
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


def control_calidad(df):
    """Motivo de exclusión por lectura ('' = válida). Todas las reglas solo miran la hora actual
    y las anteriores, así que el resultado es el mismo en tiempo real y al reconstruir el histórico."""
    d = df.sort_values(["sensor_index", "t"]).copy()
    motivo = pd.Series("", index=d.index)

    if CSV_CATALOGO.exists():
        tipo = pd.read_csv(CSV_CATALOGO).set_index("sensor_index")["location_type"]
        motivo[d["sensor_index"].map(tipo).eq(1).to_numpy()] = "interiores"

    # Ventana móvil por sensor (incluye la lectura actual)
    serie = d.set_index("t").groupby("sensor_index")["PM25_ugm3"]
    conteo = serie.rolling(VENTANA_QC).count().to_numpy()
    ceros = (d.assign(cero=d["PM25_ugm3"].eq(0).astype(float)).set_index("t")
             .groupby("sensor_index")["cero"].rolling(VENTANA_QC).mean().to_numpy())
    mediana = serie.rolling(VENTANA_QC).median().to_numpy()
    suficiente = conteo >= QC["minimo_lecturas"]
    libre = motivo.eq("").to_numpy()
    motivo[libre & suficiente & (ceros > QC["fraccion_ceros"])] = "sin_senal"
    libre = motivo.eq("").to_numpy()
    motivo[libre & suficiente & (mediana > QC["mediana_saturada"])] = "saturado"

    # Valor congelado: posición dentro de una racha de lecturas idénticas consecutivas
    racha = d.groupby("sensor_index")["PM25_ugm3"].diff().ne(0).cumsum()
    posicion = d.groupby(racha).cumcount() + 1
    motivo[motivo.eq("") & (posicion >= QC["racha_congelada"])] = "congelado"

    # Pico aislado frente a la mediana de la red en la misma hora
    mediana_hora = d.groupby("timestamp")["PM25_ugm3"].transform("median")
    pico = (d["PM25_ugm3"] > QC["factor_pico"] * mediana_hora) & (d["PM25_ugm3"] > QC["minimo_pico"])
    motivo[motivo.eq("") & pico] = "pico"
    return motivo.reindex(df.index)


def corregir_lecturas(df, motivo):
    """Lecturas que pasan el control de calidad, con PM2.5 corregido. Si la humedad del sensor
    no es válida se usa la mediana de los demás sensores en esa hora."""
    d = df.loc[motivo.eq("")].copy()
    hr_valida = d["Humedad_pct"].where(d["Humedad_pct"].between(*RANGO_HR_SENSOR))
    d["hr_usada"] = hr_valida.fillna(hr_valida.groupby(d["timestamp"]).transform("median"))
    d = d.dropna(subset=["hr_usada"])
    d["PM25_ugm3"] = correccion_epa(d["PM25_ugm3"], d["hr_usada"])
    return d


# =============================================================================
# AGEBs y censo
# =============================================================================
def leer_csv_inegi(ruta):
    """INEGI publica estos CSV en UTF-8 con BOM o en Latin-1, según el producto."""
    claves = {col: str for col in ("ENTIDAD", "MUN", "LOC", "AGEB", "MZA")}
    try:
        return pd.read_csv(ruta, dtype=claves, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(ruta, dtype=claves, encoding="latin-1", low_memory=False)


def cargar_datos_censales(ruta_csv):
    """Carga y procesa los indicadores sociodemográficos del censo INEGI por AGEB."""
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró {ruta_csv}")

    df = leer_csv_inegi(ruta_csv)
    faltantes = {"ENTIDAD", "MUN", "LOC", "AGEB"} - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en {ruta_csv.name}: {sorted(faltantes)}")

    df["AGEB"] = df["AGEB"].str.zfill(4)
    if "MZA" in df.columns:
        df = df[df["MZA"].str.zfill(3) == "000"].copy()
    df = df[df["AGEB"] != "0000"].copy()

    df["CVEGEO_CENSO"] = (
        df["ENTIDAD"].str.zfill(2) + df["MUN"].str.zfill(3) +
        df["LOC"].str.zfill(4) + df["AGEB"]
    )

    # Los valores reservados por INEGI ('*', 'N/D') quedan en 0, como en el script original.
    for col in ("POBTOT", "P_0A2", "P_3A5", "P_60YMAS", "PCON_DISC"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            print(f"Advertencia: {ruta_csv.name} no incluye la columna {col}.")
            df[col] = 0

    df["NIÑOS_0A5"] = df["P_0A2"] + df["P_3A5"]
    df = df.rename(columns={
        "POBTOT": "POBLACION_TOTAL",
        "P_60YMAS": "ADULTOS_MAYORES",
        "PCON_DISC": "PERSONAS_DISCAPACIDAD",
    })
    columnas = ["CVEGEO_CENSO", "POBLACION_TOTAL", "NIÑOS_0A5",
                "ADULTOS_MAYORES", "PERSONAS_DISCAPACIDAD"]
    return df[columnas].drop_duplicates(subset="CVEGEO_CENSO")


def clave_ageb(gdf):
    if "CVEGEO" in gdf.columns:
        return gdf["CVEGEO"].astype(str).str.zfill(13)
    partes = {"CVE_ENT": 2, "CVE_MUN": 3, "CVE_LOC": 4, "CVE_AGEB": 4}
    if set(partes) <= set(gdf.columns):
        clave = ""
        for columna, ancho in partes.items():
            clave = clave + gdf[columna].astype(str).str.zfill(ancho)
        return clave
    raise ValueError(f"El SHP no tiene CVEGEO. Columnas: {list(gdf.columns)}")


def cargar_agebs(archivo_shp, df_censo):
    """AGEBs en WGS84 con su información censal (GeoDataFrame)."""
    if not archivo_shp.exists():
        raise FileNotFoundError(f"No se encontró {archivo_shp}")
    gdf = gpd.read_file(archivo_shp)
    if gdf.crs is None:
        raise ValueError("El SHP no tiene CRS definido en su archivo .prj.")
    gdf = gdf.to_crs("EPSG:4326")
    gdf["cvegeo"] = clave_ageb(gdf)
    gdf = gdf.merge(df_censo, left_on="cvegeo", right_on="CVEGEO_CENSO", how="left")
    print(f"Cruce con el censo: {gdf['CVEGEO_CENSO'].notna().sum()} de {len(gdf)} AGEBs.")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)
    for origen, destino in (("POBLACION_TOTAL", "poblacion_total"), ("NIÑOS_0A5", "ninos_0a5"),
                            ("ADULTOS_MAYORES", "adultos_mayores"),
                            ("PERSONAS_DISCAPACIDAD", "personas_discapacidad")):
        gdf[destino] = pd.to_numeric(gdf[origen], errors="coerce").fillna(0).astype(int)
    return gdf[["cvegeo", "poblacion_total", "ninos_0a5", "adultos_mayores",
                "personas_discapacidad", "geometry"]]


# =============================================================================
# Interpolación
# =============================================================================
class Interpolador:
    """Interpolación por AGEB con el mismo método de la versión anterior: promedio de los sensores
    dentro del AGEB o interpolación lineal de Delaunay en su centroide."""

    def __init__(self, agebs):
        self.agebs = agebs
        with warnings.catch_warnings():     # centroide en grados, como la versión anterior
            warnings.simplefilter("ignore", UserWarning)
            c = agebs.geometry.centroid
        self.centroides = np.column_stack([c.x.to_numpy(), c.y.to_numpy()])

    def _sensores_por_ageb(self, puntos):
        g = gpd.GeoDataFrame(geometry=gpd.points_from_xy(puntos[:, 0], puntos[:, 1]), crs="EPSG:4326")
        unidos = gpd.sjoin(g, self.agebs[["geometry"]], how="inner", predicate="within")
        por_ageb = [[] for _ in range(len(self.agebs))]
        for i_punto, i_ageb in zip(unidos.index, unidos["index_right"]):
            por_ageb[i_ageb].append(i_punto)
        return por_ageb

    def interpolar(self, lecturas, columna):
        """Devuelve (valores, metodos, n_sensores) con un elemento por AGEB."""
        n = len(self.agebs)
        if len(lecturas) < MIN_SENSORES:
            return np.full(n, np.nan), np.array(["sin_datos"] * n, dtype=object), np.zeros(n, dtype=int)
        puntos = lecturas[["lon", "lat"]].to_numpy(dtype=float)
        valores = lecturas[columna].to_numpy(dtype=float)
        # Los sensores en las mismas coordenadas se promedian para triangular.
        unicos, inversa = np.unique(puntos, axis=0, return_inverse=True)
        inversa = inversa.ravel()
        try:
            triangulacion = Delaunay(unicos)
            por_ubicacion = np.bincount(inversa, weights=valores) / np.bincount(inversa)
            lineales = LinearNDInterpolator(triangulacion, por_ubicacion)(self.centroides)
        except Exception as error:  # Qhull falla con sensores colineales
            print(f"Error Delaunay: {error}")
            lineales = np.full(n, np.nan)

        salida = np.full(n, np.nan)
        metodos = np.array(["sin_datos"] * n, dtype=object)
        conteo = np.zeros(n, dtype=int)
        for i, (indices, lineal) in enumerate(zip(self._sensores_por_ageb(puntos), lineales)):
            if indices:
                salida[i], metodos[i], conteo[i] = round(float(np.mean(valores[indices])), 2), "promedio_sensores", len(indices)
            elif np.isfinite(lineal):
                salida[i], metodos[i] = round(float(lineal), 2), "interpolacion_delaunay"
        return salida, metodos, conteo


def procesar_hora(crudas, corregidas, interpolador):
    """Interpola una hora con y sin corrección. 'crudas' son las lecturas en rango (método anterior);
    'corregidas' las que pasaron el control de calidad, con PM2.5 corregido."""
    pm25, metodo, n_sensores = interpolador.interpolar(corregidas, "PM25_ugm3")
    pm10, _, _ = interpolador.interpolar(corregidas, "PM10_ugm3")
    pm25_crudo, _, _ = interpolador.interpolar(crudas, "PM25_ugm3")
    pm10_crudo, _, _ = interpolador.interpolar(crudas, "PM10_ugm3")
    return {"pm25": pm25, "pm10": pm10, "pm25_crudo": pm25_crudo, "pm10_crudo": pm10_crudo,
            "metodo": metodo, "n_sensores": n_sensores, "sensores_validos": len(corregidas)}


def filas_historico(agebs, timestamp, res):
    """Una fila por AGEB con algún valor (corregido o sin corregir)."""
    nulo = lambda v: None if np.isnan(v) else float(v)
    tabla = pd.DataFrame({
        "timestamp": timestamp,
        "cvegeo": agebs["cvegeo"].to_numpy(),
        "pm25_interpolado": [nulo(v) for v in res["pm25"]],
        "calidad_pm25": [clasificar_calidad_aire_pm25(nulo(v)) for v in res["pm25"]],
        "pm10_interpolado": [nulo(v) for v in res["pm10"]],
        "calidad_pm10": [clasificar_calidad_aire_pm10(nulo(v)) for v in res["pm10"]],
        "pm25_sin_correccion": [nulo(v) for v in res["pm25_crudo"]],
        "pm10_sin_correccion": [nulo(v) for v in res["pm10_crudo"]],
        "metodo": res["metodo"],
        "n_sensores": res["n_sensores"],
        "sensores_validos": res["sensores_validos"],
        "poblacion_total": agebs["poblacion_total"].to_numpy(),
        "ninos_0a5": agebs["ninos_0a5"].to_numpy(),
        "adultos_mayores": agebs["adultos_mayores"].to_numpy(),
        "personas_discapacidad": agebs["personas_discapacidad"].to_numpy(),
        "version_correccion": VERSION_CORRECCION,
    }, columns=COLUMNAS_HISTORICO)
    con_valor = tabla[["pm25_interpolado", "pm10_interpolado", "pm25_sin_correccion",
                       "pm10_sin_correccion"]].notna().any(axis=1)
    return tabla[con_valor].drop_duplicates(subset="cvegeo")


# =============================================================================
# Salidas
# =============================================================================
def generar_geojson_colonias(ruta, agebs, valores, valores_sin_correccion, clasificar, fecha):
    features = []
    for (_, fila), valor, crudo in zip(agebs.iterrows(), valores, valores_sin_correccion):
        valor = None if np.isnan(valor) else float(valor)
        crudo = None if np.isnan(crudo) else float(crudo)
        # Etiquetas descriptivas; el espacio inicial las muestra primero en los visores.
        features.append({
            "type": "Feature",
            "geometry": mapping(fila.geometry),
            "properties": {
                " Valor Interpolado": valor,
                " Calidad del Aire": clasificar(valor),
                "Valor sin corrección": crudo,
                "Corrección aplicada": VERSION_CORRECCION,
                "Población Total": int(fila["poblacion_total"]),
                "Niños de 0 a 5 años": int(fila["ninos_0a5"]),
                "Mayores de 60 años": int(fila["adultos_mayores"]),
                "Personas con Discapacidad": int(fila["personas_discapacidad"]),
                "Clave Geográfica (CVEGEO)": fila["cvegeo"],
                "Fecha de Actualización": fecha,
            },
        })

    # Escritura en temporal para no dejar la capa a medias si el proceso se interrumpe.
    tmp = ruta.with_suffix(ruta.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as archivo:
        json.dump({"type": "FeatureCollection", "features": features}, archivo,
                  ensure_ascii=False, indent=2, allow_nan=False)
    tmp.replace(ruta)
    print(f"GeoJSON generado ({len(features)} AGEBs): {ruta.name}")


def encabezado_actual(ruta):
    with open(ruta, encoding="utf-8-sig") as archivo:
        return archivo.readline().strip().split(",")


def ultimo_timestamp(ruta):
    """Lee solo el final del archivo: el histórico crece cada hora y siempre se anexa."""
    with open(ruta, "rb") as archivo:
        archivo.seek(max(0, ruta.stat().st_size - 4096))
        lineas = archivo.read().decode("utf-8", errors="ignore").strip().splitlines()
    return lineas[-1].split(",")[0] if lineas else None


def guardar_historico(tabla, timestamp, ruta=CSV_HISTORICO):
    """Agrega al CSV histórico las filas de una hora. No escribe si el archivo tiene el formato
    anterior (hay que reconstruirlo primero) o si esa hora ya es la última registrada."""
    existe = ruta.exists() and ruta.stat().st_size > 0
    if existe and encabezado_actual(ruta) != COLUMNAS_HISTORICO:
        raise ValueError(
            f"{ruta.name} tiene el formato anterior (sin corrección EPA). Ejecuta primero "
            "reconstruir_historico_interpolacion.py para convertirlo; así no se mezclan "
            "valores corregidos y sin corregir en la misma columna.")
    if existe and ultimo_timestamp(ruta) == timestamp:
        print(f"Histórico sin cambios: {timestamp} ya está en {ruta.name}")
        return
    if tabla.empty:
        print("Sin valores interpolados que guardar en el histórico.")
        return
    tabla.to_csv(ruta, mode="a", header=not existe, index=False, encoding="utf-8")
    print(f"Histórico actualizado (+{len(tabla)} filas): {ruta.name}")


# =============================================================================
# Ejecución en tiempo real
# =============================================================================
def contexto_control_calidad(lecturas):
    """Lecturas de los días previos (mismo criterio que la reconstrucción) + la hora actual."""
    historial = leer_historial_sensores()
    if historial.empty:
        return lecturas
    actual = lecturas["t"].max()
    previas = historial[(historial["t"] < actual) & (historial["t"] >= actual - pd.Timedelta(VENTANA_QC))]
    return pd.concat([previas, lecturas], ignore_index=True)


def main():
    # 1. Validar insumos antes de cargar las capas espaciales
    lecturas = leer_lecturas(CSV_PURPLEAIR)
    if len(lecturas) < MIN_SENSORES:
        print(f"Sin capas nuevas: {len(lecturas)} sensores válidos (mínimo {MIN_SENSORES}). "
              "Se conservan las capas anteriores.")
        return
    fecha = str(lecturas["timestamp"].max())
    lecturas = lecturas[lecturas["timestamp"] == fecha].reset_index(drop=True)

    # 2. Control de calidad con el historial reciente y corrección EPA
    contexto = contexto_control_calidad(lecturas)
    motivo = control_calidad(contexto)
    es_actual = contexto["timestamp"].eq(fecha)
    corregidas = corregir_lecturas(contexto[es_actual], motivo[es_actual])
    excluidas = motivo[es_actual & motivo.ne("")].value_counts()
    print(f"Control de calidad: {len(corregidas)} de {len(lecturas)} sensores válidos"
          + (f" (excluidos: {excluidas.to_dict()})" if len(excluidas) else ""))
    if len(corregidas) < MIN_SENSORES:
        print("Sin capas nuevas: muy pocos sensores pasan el control de calidad. "
              "Se conservan las capas anteriores.")
        return

    # 3. Unir AGEBs y censo, interpolar y generar las capas con la fecha de las lecturas
    agebs = cargar_agebs(ARCHIVO_SHP_COLONIAS, cargar_datos_censales(CSV_CENSO_INEGI))
    res = procesar_hora(lecturas, corregidas, Interpolador(agebs))
    generar_geojson_colonias(SALIDA_GEOJSON_PM25, agebs, res["pm25"], res["pm25_crudo"],
                             clasificar_calidad_aire_pm25, fecha)
    generar_geojson_colonias(SALIDA_GEOJSON_PM10, agebs, res["pm10"], res["pm10_crudo"],
                             clasificar_calidad_aire_pm10, fecha)

    # 4. Registrar la interpolación en el histórico
    guardar_historico(filas_historico(agebs, fecha, res), fecha)
    print(f"Proceso completado con {len(corregidas)} sensores corregidos ({VERSION_CORRECCION}).")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)