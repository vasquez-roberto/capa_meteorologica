"""Interpolación lineal de PM2.5 y PM10 por AGEB a partir de las lecturas de PurpleAir.

Colocar en la subcarpeta 'Purple Air' y ejecutar después del script que actualiza
purpleair_meteo_lecturas.csv. No consulta la API ni modifica las lecturas.

Salidas (en la subcarpeta 'Purple Air'):
    AQ_PM25_v2.geojson
    AQ_PM10.geojson
    purpleair_interpolacion_historico.csv   un registro por AGEB y lectura
"""
import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from shapely.geometry import Point, mapping


# Definición de rutas adaptadas dinámicamente a la estructura de carpetas
BASE_DIR = Path(__file__).resolve().parent  # Subcarpeta 'Purple Air'
RAIZ_PROYECTO = BASE_DIR.parent  # Carpeta raíz del proyecto

# Archivos de entrada
CSV_PURPLEAIR = BASE_DIR / "purpleair_meteo_lecturas.csv"
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

COLUMNAS_LECTURAS = ["timestamp", "sensor_index", "PM25_ugm3", "PM10_ugm3", "lat", "lon"]
COLUMNAS_HISTORICO = [
    "timestamp", "cvegeo",
    "pm25_interpolado", "calidad_pm25",
    "pm10_interpolado", "calidad_pm10",
    "metodo", "n_sensores",
    "poblacion_total", "ninos_0a5", "adultos_mayores", "personas_discapacidad",
]


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


def leer_lecturas(ruta_csv):
    """Devuelve las lecturas válidas para interpolar. 'ND' y los valores fuera de
    rango se descartan; el CSV puede estar vacío si la última consulta falló."""
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró {ruta_csv}")

    try:
        df = pd.read_csv(ruta_csv, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=COLUMNAS_LECTURAS)
    if df.empty:
        return pd.DataFrame(columns=COLUMNAS_LECTURAS)

    faltantes = set(COLUMNAS_LECTURAS) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en {ruta_csv.name}: {sorted(faltantes)}")

    total = len(df)
    for columna in ("PM25_ugm3", "PM10_ugm3", "lat", "lon"):
        df[columna] = pd.to_numeric(df[columna], errors="coerce")
    df = df.dropna(subset=["PM25_ugm3", "PM10_ugm3", "lat", "lon"])
    df = df[df["PM25_ugm3"].between(MIN_PM, MAX_PM25) & df["PM10_ugm3"].between(MIN_PM, MAX_PM10)]

    if len(df) < total:
        print(f"Lecturas descartadas por dato faltante o fuera de rango: {total - len(df)}")
    return df.reset_index(drop=True)


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


def cargar_datos_colonias_shp(archivo_shp, df_censo):
    """Lee el SHP de AGEBs, lo une con la información censal de INEGI y proyecta a WGS84."""
    if not archivo_shp.exists():
        raise FileNotFoundError(f"No se encontró {archivo_shp}")

    gdf = gpd.read_file(archivo_shp)
    if gdf.crs is None:
        raise ValueError("El SHP no tiene CRS definido en su archivo .prj.")

    gdf = gdf.to_crs("EPSG:4326")
    gdf["CVEGEO_SHP"] = clave_ageb(gdf)
    gdf = gdf.merge(df_censo, left_on="CVEGEO_SHP", right_on="CVEGEO_CENSO", how="left")
    print(f"Cruce con el censo: {gdf['CVEGEO_CENSO'].notna().sum()} de {len(gdf)} AGEBs.")

    def entero(fila, columna):
        valor = fila.get(columna)
        return int(valor) if pd.notnull(valor) else 0

    elementos = []
    for _, fila in gdf.iterrows():
        if fila.geometry is None or fila.geometry.is_empty:
            continue
        elementos.append({
            "cvegeo": str(fila["CVEGEO_SHP"]),
            "poblacion_total": entero(fila, "POBLACION_TOTAL"),
            "niños_0a5": entero(fila, "NIÑOS_0A5"),
            "adultos_mayores": entero(fila, "ADULTOS_MAYORES"),
            "personas_discapacidad": entero(fila, "PERSONAS_DISCAPACIDAD"),
            "geometry": fila.geometry,
        })
    return elementos


def preparar_interpolacion(colonias, puntos):
    """Calcula lo que comparten PM2.5 y PM10: triangulación, centroides y sensores."""
    # Los sensores en las mismas coordenadas se promedian para triangular.
    unicos, inversa = np.unique(puntos, axis=0, return_inverse=True)
    inversa = inversa.ravel()
    try:
        triangulacion = Delaunay(unicos)
    except Exception as error:  # Qhull falla con sensores colineales
        print(f"Error Delaunay: {error}")
        triangulacion = None

    centroides = np.array([[c["geometry"].centroid.x, c["geometry"].centroid.y]
                           for c in colonias], dtype=float).reshape(-1, 2)
    sensores_por_colonia = [
        [i for i, (lon, lat) in enumerate(puntos) if c["geometry"].contains(Point(lon, lat))]
        for c in colonias
    ]
    return triangulacion, inversa, centroides, sensores_por_colonia


def interpolar_valores(preparacion, valores):
    """Asigna a cada AGEB el promedio de sus sensores o, si no tiene, la interpolación
    lineal sobre la triangulación de Delaunay evaluada en su centroide. Los centroides
    fuera del área cubierta por los sensores quedan sin dato."""
    triangulacion, inversa, centroides, sensores_por_colonia = preparacion
    if triangulacion is not None:
        por_ubicacion = np.bincount(inversa, weights=valores) / np.bincount(inversa)
        lineales = LinearNDInterpolator(triangulacion, por_ubicacion)(centroides)
    else:
        lineales = np.full(len(centroides), np.nan)

    resultado = []
    for indices, lineal in zip(sensores_por_colonia, lineales):
        if indices:
            valor, metodo = round(float(np.mean(valores[indices])), 2), "promedio_sensores"
        elif np.isfinite(lineal):
            valor, metodo = round(float(lineal), 2), "interpolacion_delaunay"
        else:
            valor, metodo = None, "sin_datos"
        resultado.append({"valor": valor, "metodo": metodo, "n_sensores": len(indices)})
    return resultado


def generar_geojson_colonias(ruta, colonias, resultados, clasificar, fecha):
    features = []
    for colonia, resultado in zip(colonias, resultados):
        valor = resultado["valor"]
        # Etiquetas descriptivas; el espacio inicial las muestra primero en los visores.
        features.append({
            "type": "Feature",
            "geometry": mapping(colonia["geometry"]),
            "properties": {
                " Valor Interpolado": valor,
                " Calidad del Aire": clasificar(valor),
                "Población Total": colonia["poblacion_total"],
                "Niños de 0 a 5 años": colonia["niños_0a5"],
                "Mayores de 60 años": colonia["adultos_mayores"],
                "Personas con Discapacidad": colonia["personas_discapacidad"],
                "Clave Geográfica (CVEGEO)": colonia["cvegeo"],
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


def ultimo_timestamp(ruta):
    """Lee solo el final del archivo: el histórico crece cada hora y siempre se anexa."""
    with open(ruta, "rb") as archivo:
        archivo.seek(max(0, ruta.stat().st_size - 4096))
        lineas = archivo.read().decode("utf-8", errors="ignore").strip().splitlines()
    return lineas[-1].split(",")[0] if lineas else None


def guardar_historico(colonias, res25, res10, timestamp):
    """Agrega al CSV histórico una fila por AGEB con valor. Si ese timestamp ya es el
    último registrado no vuelve a escribirlo, para no duplicar al reejecutar con las
    mismas lecturas."""
    ruta = CSV_HISTORICO
    if ruta.exists() and ultimo_timestamp(ruta) == timestamp:
        print(f"Histórico sin cambios: {timestamp} ya está en {ruta.name}")
        return

    filas = [
        {
            "timestamp": timestamp,
            "cvegeo": colonia["cvegeo"],
            "pm25_interpolado": r25["valor"],
            "calidad_pm25": clasificar_calidad_aire_pm25(r25["valor"]),
            "pm10_interpolado": r10["valor"],
            "calidad_pm10": clasificar_calidad_aire_pm10(r10["valor"]),
            "metodo": r25["metodo"],
            "n_sensores": r25["n_sensores"],
            "poblacion_total": colonia["poblacion_total"],
            "ninos_0a5": colonia["niños_0a5"],
            "adultos_mayores": colonia["adultos_mayores"],
            "personas_discapacidad": colonia["personas_discapacidad"],
        }
        for colonia, r25, r10 in zip(colonias, res25, res10)
        if r25["valor"] is not None or r10["valor"] is not None
    ]
    tabla = pd.DataFrame(filas, columns=COLUMNAS_HISTORICO).drop_duplicates(subset="cvegeo")
    if tabla.empty:
        print("Sin valores interpolados que guardar en el histórico.")
        return

    escribir_encabezado = not ruta.exists() or ruta.stat().st_size == 0
    tabla.to_csv(ruta, mode="a", header=escribir_encabezado, index=False, encoding="utf-8")
    print(f"Histórico actualizado (+{len(tabla)} filas): {ruta.name}")


def main():
    # 1. Validar insumos antes de cargar las capas espaciales
    lecturas = leer_lecturas(CSV_PURPLEAIR)
    if len(lecturas) < MIN_SENSORES:
        print(f"Sin capas nuevas: {len(lecturas)} sensores válidos (mínimo {MIN_SENSORES}). "
              "Se conservan las capas anteriores.")
        return

    # 2. Unir AGEBs y censo
    colonias = cargar_datos_colonias_shp(ARCHIVO_SHP_COLONIAS,
                                         cargar_datos_censales(CSV_CENSO_INEGI))

    # 3. Interpolar y generar las capas con la fecha de las lecturas
    puntos = lecturas[["lon", "lat"]].to_numpy(dtype=float)
    fecha = str(lecturas["timestamp"].max())
    preparacion = preparar_interpolacion(colonias, puntos)

    res25 = interpolar_valores(preparacion, lecturas["PM25_ugm3"].to_numpy(dtype=float))
    res10 = interpolar_valores(preparacion, lecturas["PM10_ugm3"].to_numpy(dtype=float))
    generar_geojson_colonias(SALIDA_GEOJSON_PM25, colonias, res25,
                             clasificar_calidad_aire_pm25, fecha)
    generar_geojson_colonias(SALIDA_GEOJSON_PM10, colonias, res10,
                             clasificar_calidad_aire_pm10, fecha)

    # 4. Registrar la interpolación en el histórico
    guardar_historico(colonias, res25, res10, fecha)
    print(f"Proceso completado con {len(lecturas)} sensores.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)