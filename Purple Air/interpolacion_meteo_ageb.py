from datetime import datetime
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from shapely.geometry import Point, mapping

# Definición de rutas adaptadas dinámicamente a la estructura de carpetas
BASE_DIR = Path(__file__).resolve().parent  # Subcarpeta 'Purple Air'
RAIZ_PROYECTO = BASE_DIR.parent  # Carpeta raíz del proyecto

# Archivos de entrada
CSV_PURPLEAIR = BASE_DIR / "purpleair_meteo_lecturas.csv"
CSV_CENSO_INEGI = RAIZ_PROYECTO / "shp" / "cpv2020.csv"
ARCHIVO_SHP_COLONIAS = RAIZ_PROYECTO / "shp" / "2025_1_19_A.shp"

# Archivos de salida (Garantizados dentro de 'Purple Air/')
SALIDA_TEMP_GEOJSON = BASE_DIR / "AQ_Temperatura_AGEB.geojson"
SALIDA_RH_GEOJSON = BASE_DIR / "AQ_Humedad_AGEB.geojson"
CSV_HISTORICO_INTERP = BASE_DIR / "interpolacion_meteo_historico.csv"


def cargar_datos_censales(ruta_csv):
  """Carga y limpia el censo INEGI estandarizando la clave CVEGEO a 13 dígitos."""
  if not ruta_csv.exists():
    print(f"Aviso: No se encontró el censo en {ruta_csv}")
    return pd.DataFrame()

  try:
    df = pd.read_csv(ruta_csv, dtype=str, low_memory=False)

    # Filtrar nivel AGEB urbano (MZA == '000' y AGEB != '0000')
    if "MZA" in df.columns:
      df = df[df["MZA"].str.zfill(3) == "000"].copy()
    if "AGEB" in df.columns:
      df = df[df["AGEB"] != "0000"].copy()

    # Construir CVEGEO estandarizada a 13 dígitos (ENTIDAD + MUN + LOC + AGEB)
    ent = df["ENTIDAD"].astype(str).str.zfill(2)
    mun = df["MUN"].astype(str).str.zfill(3)
    loc = df["LOC"].astype(str).str.zfill(4)
    ageb = df["AGEB"].astype(str).str.zfill(4)

    df["CVEGEO_CENSO"] = ent + mun + loc + ageb

    # Extraer población total
    if "POBTOT" in df.columns:
      df["POBLACION_TOTAL"] = (
          pd.to_numeric(df["POBTOT"], errors="coerce").fillna(0).astype(int)
      )
    else:
      df["POBLACION_TOTAL"] = 0

    df_limpio = (
        df[["CVEGEO_CENSO", "POBLACION_TOTAL"]]
        .drop_duplicates(subset=["CVEGEO_CENSO"])
        .copy()
    )
    print(
        f"Éxito: Censo INEGI procesado ({len(df_limpio)} AGEBs vinculados a 13"
        " dígitos)."
    )
    return df_limpio

  except Exception as e:
    print(f"Error cargando censo INEGI: {e}")
    return pd.DataFrame()


def cargar_colonias(archivo_shp, df_censo):
  """Carga la capa de AGEBs (SHP), proyecta a WGS84 y cruza los datos censales."""
  gdf = gpd.read_file(archivo_shp)
  if gdf.crs is None:
    raise ValueError("El SHP no tiene CRS definido.")
  gdf = gdf.to_crs("EPSG:4326")

  posibles_llaves = ["CVEGEO", "CVE_AGEB", "CODIGO"]
  col_cve = next(
      (col for col in posibles_llaves if col in gdf.columns), gdf.columns[0]
  )
  gdf["CVEGEO_SHP"] = gdf[col_cve].astype(str).str.zfill(13)

  if not df_censo.empty:
    gdf = gdf.merge(
        df_censo, left_on="CVEGEO_SHP", right_on="CVEGEO_CENSO", how="left"
    )

  elementos = []
  for _, fila in gdf.iterrows():
    if fila.geometry is not None and not fila.geometry.is_empty:
      elementos.append({
          "cvegeo": str(fila["CVEGEO_SHP"]),
          "poblacion_total": (
              int(fila.get("POBLACION_TOTAL", 0))
              if pd.notnull(fila.get("POBLACION_TOTAL"))
              else 0
          ),
          "geometry": fila.geometry,
      })
  return elementos


def interpolar_lineal(punto, triangulo_indices, puntos, valores):
  """Calcula la interpolación lineal plana dentro del triángulo de Delaunay."""
  v0, v1, v2 = puntos[triangulo_indices]
  z0, z1, z2 = valores[triangulo_indices]
  try:
    pesos = np.linalg.solve(np.array([v1 - v0, v2 - v0]).T, punto - v0)
    return (1 - pesos[0] - pesos[1]) * z0 + pesos[0] * z1 + pesos[1] * z2
  except np.linalg.LinAlgError:
    return None


def procesar_interpolaciones(colonias_data, puntos_data, temps, hums, timestamp):
  """Ejecuta la interpolación espacial, genera GeoJSONs numéricos y guarda el histórico."""
  try:
    triangulacion = Delaunay(puntos_data)
  except Exception as e:
    print(f"Error generando Triangulación de Delaunay: {e}")
    triangulacion = None

  features_temp, features_rh, filas_historico = [], [], []

  for colonia in colonias_data:
    geom = colonia["geometry"]
    cvegeo = colonia["cvegeo"]
    pob = colonia["poblacion_total"]

    # 1. Verificar si hay sensores dentro del polígono del AGEB
    indices_dentro = [
        i
        for i, (lon, lat) in enumerate(puntos_data)
        if geom.contains(Point(lon, lat))
    ]

    if indices_dentro:
      v_temp = float(np.mean([temps[i] for i in indices_dentro]))
      v_rh = float(np.mean([hums[i] for i in indices_dentro]))
    elif triangulacion is None:
      v_temp = v_rh = None
    else:
      # 2. Interpolación lineal en el triángulo de Delaunay usando el centroide
      centroide = geom.centroid
      punto = np.array([centroide.x, centroide.y])
      indice = triangulacion.find_simplex(punto)
      if indice != -1:
        v_temp = interpolar_lineal(
            punto, triangulacion.simplices[indice], puntos_data, temps
        )
        v_rh = interpolar_lineal(
            punto, triangulacion.simplices[indice], puntos_data, hums
        )
      else:
        v_temp = v_rh = None

    # Mantenemos valores flotantes puros (None en Python -> null en JSON) para habilitar rampas de color en ArcGIS
    v_temp_val = (
        round(float(v_temp), 2)
        if v_temp is not None and np.isfinite(v_temp)
        else None
    )
    v_rh_val = (
        round(float(v_rh), 2) if v_rh is not None and np.isfinite(v_rh) else None
    )

    # GeoJSON Temperatura
    features_temp.append({
        "type": "Feature",
        "geometry": mapping(geom),
        "properties": {
            "Temperatura_C": v_temp_val,
            "Poblacion Total": pob,
            "Clave Geografica (CVEGEO)": cvegeo,
            "Fecha": timestamp,
            "Fuente": "Interpolacion Delaunay PurpleAir",
        },
    })

    # GeoJSON Humedad
    features_rh.append({
        "type": "Feature",
        "geometry": mapping(geom),
        "properties": {
            "Humedad_Relativa_pct": v_rh_val,
            "Poblacion Total": pob,
            "Clave Geografica (CVEGEO)": cvegeo,
            "Fecha": timestamp,
            "Fuente": "Interpolacion Delaunay PurpleAir",
        },
    })

    # Fila para el CSV Histórico
    filas_historico.append({
        "timestamp": timestamp,
        "cvegeo": cvegeo,
        "temperatura_interpolada_C": v_temp_val,
        "humedad_interpolada_pct": v_rh_val,
        "poblacion_total": pob,
    })

  # Exportar archivos GeoJSON
  with open(SALIDA_TEMP_GEOJSON, "w", encoding="utf-8") as f:
    json.dump(
        {"type": "FeatureCollection", "features": features_temp},
        f,
        ensure_ascii=False,
        indent=2,
    )

  with open(SALIDA_RH_GEOJSON, "w", encoding="utf-8") as f:
    json.dump(
        {"type": "FeatureCollection", "features": features_rh},
        f,
        ensure_ascii=False,
        indent=2,
    )

  # Exportar / Anexar al CSV Histórico
  df_hist = pd.DataFrame(filas_historico)
  if not CSV_HISTORICO_INTERP.exists():
    df_hist.to_csv(CSV_HISTORICO_INTERP, index=False, encoding="utf-8-sig")
  else:
    df_hist.to_csv(
        CSV_HISTORICO_INTERP,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8-sig",
    )

  print(
      "Éxito final: GeoJSONs generados y guardados en"
      f" '{CSV_HISTORICO_INTERP.name}'."
  )


if __name__ == "__main__":
  if not CSV_PURPLEAIR.exists():
    print(
        f"Error: No se encontró {CSV_PURPLEAIR}. Ejecuta primero"
        " 'purpleair_meteo.py'."
    )
    exit()

  timestamp_actual = datetime.now().strftime("%Y-%m-%d %H:00:00")
  df_purple = pd.read_csv(CSV_PURPLEAIR)

  # Normalización de coordenadas (soporta 'lat'/'lon' o 'latitude'/'longitude')
  col_lat = "lat" if "lat" in df_purple.columns else "latitude"
  col_lon = "lon" if "lon" in df_purple.columns else "longitude"

  # Convertir lecturas a formato numérico
  df_purple["Temperatura_C"] = pd.to_numeric(
      df_purple["Temperatura_C"], errors="coerce"
  )
  df_purple["Humedad_pct"] = pd.to_numeric(
      df_purple["Humedad_pct"], errors="coerce"
  )

  df_purple = df_purple.dropna(
      subset=[col_lat, col_lon, "Temperatura_C", "Humedad_pct"]
  )

  puntos = df_purple[[col_lon, col_lat]].to_numpy()
  temps = df_purple["Temperatura_C"].to_numpy()
  hums = df_purple["Humedad_pct"].to_numpy()

  if len(puntos) >= 3 and ARCHIVO_SHP_COLONIAS.exists():
    df_censo = cargar_datos_censales(CSV_CENSO_INEGI)
    colonias = cargar_colonias(ARCHIVO_SHP_COLONIAS, df_censo)
    procesar_interpolaciones(colonias, puntos, temps, hums, timestamp_actual)
  else:
    print(
        f"Insuficientes puntos ({len(puntos)}) o no existe SHP en:"
        f" {ARCHIVO_SHP_COLONIAS}"
    )