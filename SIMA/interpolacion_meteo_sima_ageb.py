from datetime import datetime
import json
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from shapely.geometry import Point

# Rutas dinámicas
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

CSV_SIMA_LECTURAS = BASE_DIR / "sima_meteo_lecturas.csv"
SHP_AGEB = ROOT_DIR / "shp" / "2025_1_19_A.shp"

CSV_INTERPOLACION_ACTUAL = BASE_DIR / "sima_interpolacion_lecturas.csv"
CSV_INTERPOLACION_HISTORICO = BASE_DIR / "sima_interpolacion_historico.csv"

GEOJSON_TEMP = BASE_DIR / "sima_Temperatura_AGEB.geojson"
GEOJSON_HUM = BASE_DIR / "sima_Humedad_AGEB.geojson"
GEOJSON_RAD = BASE_DIR / "sima_Radiacion_AGEB.geojson"


def ejecutar_kriging_ordinario(
    lons, lats, valores, target_lons, target_lats, modelo_variograma="linear"
):
  """Ejecuta Kriging Ordinario y retorna la estimación y la varianza del error."""
  try:
    # Construcción y ajuste del modelo Geoestadístico
    OK = OrdinaryKriging(
        lons,
        lats,
        valores,
        variogram_model=modelo_variograma,
        verbose=False,
        enable_plotting=False,
        exact_values=True,
    )

    # Evaluación puntual sobre los centroides de las AGEBs
    z_est, ss_var = OK.execute("points", target_lons, target_lats)
    return np.round(z_est.data, 3), np.round(ss_var.data, 4)
  except Exception as e:
    print(f"Error al ejecutar Kriging: {e}")
    # Retornar NaNs si hay fallo de convergencia por matriz singular
    return np.full(len(target_lons), np.nan), np.full(len(target_lons), np.nan)


print("Iniciando Interpolación Geoestadística (Kriging Ordinario) para SIMA...")

# 1. Cargar lecturas del SIMA
if not CSV_SIMA_LECTURAS.exists():
  raise FileNotFoundError(f"No se encontró {CSV_SIMA_LECTURAS}")

df_sima = pd.read_csv(CSV_SIMA_LECTURAS)

for col in ["Temperatura_C", "Humedad_pct", "RadiacionSolar_Wm2"]:
  df_sima[col] = pd.to_numeric(
      df_sima[col].replace("ND", np.nan), errors="coerce"
  )

# 2. Cargar Shapefile AGEB
if not SHP_AGEB.exists():
  raise FileNotFoundError(f"No se encontró el shapefile {SHP_AGEB}")

gdf_ageb = gpd.read_file(SHP_AGEB)
if gdf_ageb.crs is None or gdf_ageb.crs.to_epsg() != 4326:
  gdf_ageb = gdf_ageb.to_crs(epsg=4326)

centroides = gdf_ageb.geometry.centroid
target_lons = centroides.x.values
target_lats = centroides.y.values

timestamp_str = (
    df_sima["timestamp"].iloc[0]
    if "timestamp" in df_sima.columns
    else datetime.now().strftime("%Y-%m-%d %H:00:00")
)

# 3. Kriging Ordinario para Temperatura
df_temp = df_sima.dropna(subset=["Temperatura_C", "lat", "lon"])
if len(df_temp) >= 3:
  z_temp, var_temp = ejecutar_kriging_ordinario(
      df_temp["lon"].values,
      df_temp["lat"].values,
      df_temp["Temperatura_C"].values,
      target_lons,
      target_lats,
      modelo_variograma="spherical",
  )
  gdf_ageb["Temperatura_C"] = z_temp
  gdf_ageb["Var_Temp"] = var_temp
else:
  gdf_ageb["Temperatura_C"] = np.nan
  gdf_ageb["Var_Temp"] = np.nan

# 4. Kriging Ordinario para Humedad Relativa
df_hum = df_sima.dropna(subset=["Humedad_pct", "lat", "lon"])
if len(df_hum) >= 3:
  z_hum, var_hum = ejecutar_kriging_ordinario(
      df_hum["lon"].values,
      df_hum["lat"].values,
      df_hum["Humedad_pct"].values,
      target_lons,
      target_lats,
      modelo_variograma="spherical",
  )
  gdf_ageb["Humedad_pct"] = z_hum
  gdf_ageb["Var_Hum"] = var_hum
else:
  gdf_ageb["Humedad_pct"] = np.nan
  gdf_ageb["Var_Hum"] = np.nan

# 5. Kriging Ordinario para Radiación Solar
df_rad = df_sima.dropna(subset=["RadiacionSolar_Wm2", "lat", "lon"])
if len(df_rad) >= 3:
  z_rad, var_rad = ejecutar_kriging_ordinario(
      df_rad["lon"].values,
      df_rad["lat"].values,
      df_rad["RadiacionSolar_Wm2"].values,
      target_lons,
      target_lats,
      modelo_variograma="linear",
  )
  gdf_ageb["RadiacionSolar_Wm2"] = z_rad
  gdf_ageb["Var_Rad"] = var_rad
else:
  gdf_ageb["RadiacionSolar_Wm2"] = np.nan
  gdf_ageb["Var_Rad"] = np.nan

# 6. Exportar Archivos CSV
col_ageb = "CVEGEO" if "CVEGEO" in gdf_ageb.columns else gdf_ageb.columns[0]

df_resumen = pd.DataFrame({
    "timestamp": timestamp_str,
    "CVEGEO": gdf_ageb[col_ageb],
    "Temperatura_C": gdf_ageb["Temperatura_C"].fillna("ND"),
    "Varianza_Temp": gdf_ageb["Var_Temp"].fillna("ND"),
    "Humedad_pct": gdf_ageb["Humedad_pct"].fillna("ND"),
    "Varianza_Hum": gdf_ageb["Var_Hum"].fillna("ND"),
    "RadiacionSolar_Wm2": gdf_ageb["RadiacionSolar_Wm2"].fillna("ND"),
    "Varianza_Rad": gdf_ageb["Var_Rad"].fillna("ND"),
})

df_resumen.to_csv(CSV_INTERPOLACION_ACTUAL, index=False, encoding="utf-8-sig")

if not CSV_INTERPOLACION_HISTORICO.exists():
  df_resumen.to_csv(
      CSV_INTERPOLACION_HISTORICO, index=False, encoding="utf-8-sig"
  )
else:
  df_resumen.to_csv(
      CSV_INTERPOLACION_HISTORICO,
      mode="a",
      header=False,
      index=False,
      encoding="utf-8-sig",
  )

# 7. Exportar GeoJSONs para ArcGIS Online (Incluyen Varianza e Incertidumbre)

# GeoJSON Temperatura
gdf_temp_exp = gdf_ageb[
    [col_ageb, "Temperatura_C", "Var_Temp", "geometry"]
].copy()
gdf_temp_exp["Fecha"] = timestamp_str
gdf_temp_exp["Metodo"] = "Kriging Ordinario"
gdf_temp_exp.to_file(GEOJSON_TEMP, driver="GeoJSON")

# GeoJSON Humedad
gdf_hum_exp = gdf_ageb[
    [col_ageb, "Humedad_pct", "Var_Hum", "geometry"]
].copy()
gdf_hum_exp["Fecha"] = timestamp_str
gdf_hum_exp["Metodo"] = "Kriging Ordinario"
gdf_hum_exp.to_file(GEOJSON_HUM, driver="GeoJSON")

# GeoJSON Radiación Solar
gdf_rad_exp = gdf_ageb[
    [col_ageb, "RadiacionSolar_Wm2", "Var_Rad", "geometry"]
].copy()
gdf_rad_exp["Fecha"] = timestamp_str
gdf_rad_exp["Metodo"] = "Kriging Ordinario"
gdf_rad_exp.to_file(GEOJSON_RAD, driver="GeoJSON")

print("Interpolación geoestadística (Kriging) finalizada con éxito.")