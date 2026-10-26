"""
=============================================================================
INTERPOLACIÓN ESPACIAL DE LA RED SIMA (15 ESTACIONES) A AGEBs DEL INEGI
=============================================================================
Variables: PM10, PM2.5, Temperatura, Humedad Relativa, Radiación Solar.

POR QUÉ FALLABA EL KRIGING ORDINARIO
------------------------------------
El Kriging necesita primero ESTIMAR un variograma (cómo cambia la diferencia
entre dos estaciones conforme se alejan). Para eso se usan los pares de
estaciones: con 15 estaciones hay solo 105 pares, y se reparten en ~6 "lags".
Eso es muy poco para ajustar sill, rango y nugget de forma estable. Resultado:
  - el variograma ajustado es casi arbitrario (a veces plano o con rango absurdo)
  - la varianza de kriging sale enorme (depende del variograma, no de los datos)
  - con exact_values=True, el mapa pasa por cada estación y crea "picos"
Además, la temperatura en Monterrey depende de la altitud/urbanización
(Sierra Madre, Cerro de la Silla): hay tendencia, no es un campo estacionario.

QUÉ HACE ESTE SCRIPT EN SU LUGAR
--------------------------------
En vez de apostar por UN método, prueba varios y deja que los DATOS decidan:

  1. IDW (ponderación por inverso de la distancia), con varias potencias.
  2. RBF "thin-plate spline" (superficie suave) con varios suavizados.
  3. Kriging Ordinario (opcional, si hay pykrige) con distintos variogramas.
  4. "Media" (el valor promedio de la red, sin geografía) como línea base.

Para cada variable se hace VALIDACIÓN CRUZADA "LEAVE-ONE-OUT" (LOOCV):
se quita una estación, se predice su valor con las otras 14 y se compara con
lo que realmente midió. Se repite con las 15. El método con menor error (RMSE)
es el que se usa para esa variable y ese momento.

  - Si ningún método le gana a la Media, el script lo dice y usa la Media:
    significa que con estas estaciones no hay estructura espacial confiable.

INCERTIDUMBRE (honesta, sin varianza de kriging)
------------------------------------------------
Cada AGEB lleva dos columnas:
  - Error_CV      : RMSE de validación cruzada (error típico esperado, en las
                    unidades de la variable). Es el mismo para toda la variable.
  - Dist_Est_km   : distancia a la estación más cercana. Mientras mayor sea,
                    menos confiable es la estimación en ese AGEB.

ÁREA DE ESTUDIO
---------------
Solo se estiman los AGEBs cuyo punto representativo cae dentro del polígono
que envuelve a las 15 estaciones (casco convexo) + un pequeño buffer
(BUFFER_KM). Fuera de ahí NO se extrapola.

Uso:
    python sima_interpolacion.py            # normal
    python sima_interpolacion.py --ranking  # imprime el ranking de métodos
=============================================================================
"""
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist

try:
    from pykrige.ok import OrdinaryKriging
    HAY_PYKRIGE = True
except ImportError:  # el script funciona sin pykrige
    HAY_PYKRIGE = False

warnings.filterwarnings("ignore")
MOSTRAR_RANKING = "--ranking" in sys.argv

# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

CSV_LECTURAS = BASE_DIR / "sima_meteo_lecturas.csv"
SHP_AGEB = ROOT_DIR / "shp" / "2025_1_19_A.shp"

CSV_ACTUAL = BASE_DIR / "sima_interpolacion_lecturas.csv"
CSV_HISTORICO = BASE_DIR / "sima_interpolacion_historico.csv"
CSV_DIAGNOSTICO = BASE_DIR / "sima_interpolacion_diagnostico.csv"

# UTM 14N: sistema en METROS. Las distancias deben calcularse en metros/km,
# NO en grados (un grado de longitud y de latitud no miden lo mismo).
CRS_METRICO = 32614
BUFFER_KM = 2.0          # margen alrededor del polígono de estaciones
MIN_ESTACIONES = 4       # mínimo de estaciones con dato para interpolar
MARGEN_RANGO = 0.15      # se permite salir 15% del rango observado (evita picos)

# col: columna del CSV de lecturas | salida: nombre del GeoJSON
# lim_min / lim_max: límites FÍSICOS (una concentración no puede ser negativa,
# la humedad relativa no pasa de 100%).
VARIABLES = {
    "PM10": dict(col="PM10_ugm3", salida="sima_PM10_AGEB.geojson",
                 lim_min=0.0, lim_max=None),
    "PM25": dict(col="PM25_ugm3", salida="sima_PM25_AGEB.geojson",
                 lim_min=0.0, lim_max=None),
    "Temperatura": dict(col="Temperatura_C", salida="sima_Temperatura_AGEB.geojson",
                        lim_min=None, lim_max=None),
    "Humedad": dict(col="Humedad_pct", salida="sima_Humedad_AGEB.geojson",
                    lim_min=0.0, lim_max=100.0),
    "Radiacion": dict(col="RadiacionSolar_Wm2", salida="sima_Radiacion_AGEB.geojson",
                      lim_min=0.0, lim_max=None),
}


# =============================================================================
# 2. MÉTODOS DE INTERPOLACIÓN
# Todos tienen la misma firma: f(xy_entrenamiento, z_entrenamiento, xy_destino, **parámetros)
#   xy_* : arreglo (n, 2) con coordenadas en km
#   z_*  : arreglo (n,) con los valores medidos
# y regresan los valores estimados en xy_destino.
# =============================================================================
def metodo_media(xy_tr, z_tr, xy_pr):
    """Línea base: todos los puntos reciben el promedio de las estaciones.
    Si un método no supera a este, no está aportando información espacial."""
    return np.full(len(xy_pr), z_tr.mean())


def metodo_idw(xy_tr, z_tr, xy_pr, power):
    """IDW: el valor en un punto es un promedio de las estaciones, donde cada
    una pesa 1 / distancia^power.
      power bajo (1)  -> superficie muy suave, se parece a la media
      power alto (4)  -> domina la estación más cercana (casi 'polígonos')
    Es exacto en las estaciones y NUNCA sale del rango de los datos."""
    d = np.maximum(cdist(xy_pr, xy_tr), 1e-6)    # evita división entre 0
    w = 1.0 / d ** power
    return (w * z_tr).sum(axis=1) / w.sum(axis=1)


def metodo_rbf_tps(xy_tr, z_tr, xy_pr, smoothing):
    """Thin-plate spline: la superficie 'más lisa' que pasa cerca de los puntos
    (como una lámina flexible). Incluye una tendencia lineal este-oeste/norte-sur.
      smoothing = 0   -> pasa exacto por las estaciones (puede ondular)
      smoothing alto  -> se relaja y suaviza (tolera el ruido de la medición)
    Los valores se estandarizan para que 'smoothing' signifique lo mismo en
    cualquier variable."""
    mu, sd = z_tr.mean(), z_tr.std()
    if sd == 0:
        return np.full(len(xy_pr), mu)
    f = RBFInterpolator(xy_tr, (z_tr - mu) / sd, kernel="thin_plate_spline",
                        smoothing=smoothing, degree=1)
    return f(xy_pr) * sd + mu


def metodo_kriging(xy_tr, z_tr, xy_pr, modelo):
    """Kriging ordinario con variograma ajustado automáticamente (pykrige).
    Se trabaja en km para que el variograma tenga sentido físico."""
    ok = OrdinaryKriging(xy_tr[:, 0], xy_tr[:, 1], z_tr,
                         variogram_model=modelo, nlags=6, weight=True,
                         verbose=False, enable_plotting=False)
    z, _ = ok.execute("points", xy_pr[:, 0], xy_pr[:, 1])
    return np.asarray(z)


def construir_candidatos():
    """Lista de (nombre, función, parámetros) que competirán entre sí."""
    c = [("Media", metodo_media, {})]
    for p in (1.0, 1.5, 2.0, 3.0, 4.0):
        c.append((f"IDW p={p}", metodo_idw, {"power": p}))
    for s in (0.0, 0.05, 0.2, 1.0, 5.0):
        c.append((f"TPS suav={s}", metodo_rbf_tps, {"smoothing": s}))
    if HAY_PYKRIGE:
        for m in ("spherical", "exponential", "linear"):
            c.append((f"Kriging {m}", metodo_kriging, {"modelo": m}))
    return c


# =============================================================================
# 3. UTILIDADES: acotar valores y validación cruzada
# =============================================================================
def acotar(pred, z_tr, lim_min, lim_max):
    """Evita valores absurdos:
      - no salir más de MARGEN_RANGO del rango observado (controla sobreoscilación)
      - respetar límites físicos (no negativos, humedad <= 100)"""
    rango = z_tr.max() - z_tr.min()
    lo = z_tr.min() - MARGEN_RANGO * rango
    hi = z_tr.max() + MARGEN_RANGO * rango
    if lim_min is not None:
        lo = max(lo, lim_min)
    if lim_max is not None:
        hi = min(hi, lim_max)
    return np.clip(pred, lo, hi)


def predecir(func, params, xy_tr, z_tr, xy_pr, lim_min, lim_max):
    return acotar(func(xy_tr, z_tr, xy_pr, **params), z_tr, lim_min, lim_max)


def loocv(func, params, xy, z, lim_min, lim_max):
    """Leave-One-Out: predice cada estación usando solo las demás."""
    n = len(z)
    pred = np.full(n, np.nan)
    for i in range(n):
        m = np.arange(n) != i
        try:
            pred[i] = predecir(func, params, xy[m], z[m], xy[i:i + 1],
                               lim_min, lim_max)[0]
        except Exception:
            return None            # el método falló -> queda descartado
    if not np.all(np.isfinite(pred)):
        return None
    return pred


def seleccionar_metodo(xy, z, lim_min, lim_max):
    """Corre LOOCV para todos los candidatos y devuelve el ranking (DataFrame)."""
    filas = []
    for nombre, func, params in construir_candidatos():
        pred = loocv(func, params, xy, z, lim_min, lim_max)
        if pred is None:
            continue
        err = pred - z
        filas.append(dict(metodo=nombre, func=func, params=params,
                          rmse=float(np.sqrt(np.mean(err ** 2))),
                          mae=float(np.mean(np.abs(err))),
                          sesgo=float(np.mean(err))))
    rk = pd.DataFrame(filas).sort_values("rmse").reset_index(drop=True)
    rmse_media = float(rk.loc[rk["metodo"] == "Media", "rmse"].iloc[0])
    # Skill: 1 = perfecto, 0 = igual que usar la media, <0 = peor que la media
    rk["skill"] = 1.0 - rk["rmse"] / rmse_media if rmse_media > 0 else 0.0
    return rk


# =============================================================================
# 4. CARGA DE DATOS
# =============================================================================
print("Interpolación SIMA -> AGEB (selección de método por validación cruzada)")

if not CSV_LECTURAS.exists():
    raise FileNotFoundError(f"No se encontró {CSV_LECTURAS}")
if not SHP_AGEB.exists():
    raise FileNotFoundError(f"No se encontró {SHP_AGEB}")

df = pd.read_csv(CSV_LECTURAS, encoding="utf-8-sig", na_values=["ND"])
for v in VARIABLES.values():
    df[v["col"]] = pd.to_numeric(df[v["col"]], errors="coerce")
timestamp = str(df["timestamp"].iloc[0])

# Estaciones -> UTM (metros) -> km
est = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326
).to_crs(CRS_METRICO)
est_xy = np.column_stack([est.geometry.x, est.geometry.y]) / 1000.0

# AGEBs -> UTM; usamos un punto representativo (siempre cae dentro del polígono,
# a diferencia del centroide en AGEBs con forma irregular)
ageb = gpd.read_file(SHP_AGEB)
ageb = (ageb.set_crs(4326) if ageb.crs is None else ageb).to_crs(CRS_METRICO)
col_id = "CVEGEO" if "CVEGEO" in ageb.columns else ageb.columns[0]

# =============================================================================
# 5. ÁREA DE ESTUDIO: casco convexo de las estaciones + buffer
# =============================================================================
hull = est.geometry.union_all().convex_hull.buffer(BUFFER_KM * 1000) \
    if hasattr(est.geometry, "union_all") \
    else est.geometry.unary_union.convex_hull.buffer(BUFFER_KM * 1000)

puntos = ageb.geometry.representative_point()
dentro = puntos.within(hull)
ageb = ageb.loc[dentro].copy()
puntos = puntos.loc[dentro]
destino_xy = np.column_stack([puntos.x, puntos.y]) / 1000.0
print(f"AGEBs dentro del área de estaciones: {len(ageb)}")

# Distancia de cada AGEB a la estación más cercana (km)
ageb["Dist_Est_km"] = np.round(cdist(destino_xy, est_xy).min(axis=1), 2)

# =============================================================================
# 6. INTERPOLACIÓN VARIABLE POR VARIABLE
# =============================================================================
diagnostico = []
resultados = {}

for nombre, cfg in VARIABLES.items():
    col = cfg["col"]
    ok = df[col].notna().values
    n_ok = int(ok.sum())
    print(f"\n=== {nombre} ({n_ok} estaciones con dato) ===")

    if n_ok < MIN_ESTACIONES:
        print(f"  Se necesitan >= {MIN_ESTACIONES} estaciones. Se omite.")
        resultados[nombre] = None
        continue

    xy, z = est_xy[ok], df.loc[ok, col].values.astype(float)

    # Caso especial: todas las estaciones iguales (p. ej. radiación = 0 de noche)
    if np.isclose(z.std(), 0):
        print(f"  Valor uniforme ({z[0]:.3f}). Se asigna constante.")
        resultados[nombre] = dict(valor=np.full(len(ageb), z[0]), rmse=0.0,
                                  metodo="Constante")
        diagnostico.append(dict(timestamp=timestamp, variable=nombre,
                                metodo="Constante", rmse_cv=0.0, skill=np.nan,
                                n_estaciones=n_ok))
        continue

    ranking = seleccionar_metodo(xy, z, cfg["lim_min"], cfg["lim_max"])
    if MOSTRAR_RANKING:
        print(ranking[["metodo", "rmse", "mae", "sesgo", "skill"]]
              .round(3).to_string(index=False))

    mejor = ranking.iloc[0]
    print(f"  Método elegido : {mejor['metodo']}")
    print(f"  RMSE (LOOCV)   : {mejor['rmse']:.3f}   "
          f"| skill vs. media: {mejor['skill']:.2f}")
    if mejor["metodo"] == "Media":
        print("  AVISO: ningún método supera a la media; sin estructura espacial "
              "confiable en este momento.")

    valores = predecir(mejor["func"], mejor["params"], xy, z, destino_xy,
                       cfg["lim_min"], cfg["lim_max"])
    resultados[nombre] = dict(valor=np.round(valores, 3),
                              rmse=round(mejor["rmse"], 3),
                              metodo=mejor["metodo"])
    diagnostico.append(dict(timestamp=timestamp, variable=nombre,
                            metodo=mejor["metodo"],
                            rmse_cv=round(mejor["rmse"], 3),
                            skill=round(mejor["skill"], 3),
                            n_estaciones=n_ok))

# =============================================================================
# 7. EXPORTACIÓN: GeoJSON por variable (WGS84, listo para ArcGIS Online)
# =============================================================================
ageb_wgs = ageb.to_crs(4326)

for nombre, cfg in VARIABLES.items():
    res = resultados.get(nombre)
    if res is None:
        continue
    g = ageb_wgs[[col_id, "Dist_Est_km", "geometry"]].copy()
    g[cfg["col"]] = res["valor"]
    g["Error_CV"] = res["rmse"]
    g["Metodo"] = res["metodo"]
    g["Fecha"] = timestamp
    g = g[[col_id, cfg["col"], "Error_CV", "Dist_Est_km", "Metodo", "Fecha",
           "geometry"]]
    # GeoJSON válido: NaN -> null
    for c in (cfg["col"], "Error_CV", "Dist_Est_km"):
        g[c] = g[c].astype(object).where(g[c].notna(), None)
    g.to_file(BASE_DIR / cfg["salida"], driver="GeoJSON")

# CSV ancho (una fila por AGEB) + histórico sin duplicados
tabla = pd.DataFrame({"timestamp": timestamp, "CVEGEO": ageb[col_id].values,
                      "Dist_Est_km": ageb["Dist_Est_km"].values})
for nombre, cfg in VARIABLES.items():
    res = resultados.get(nombre)
    tabla[cfg["col"]] = res["valor"] if res else np.nan
    tabla[f"Error_CV_{nombre}"] = res["rmse"] if res else np.nan
tabla_csv = tabla.fillna("ND")
tabla_csv.to_csv(CSV_ACTUAL, index=False, encoding="utf-8-sig")

if CSV_HISTORICO.exists():
    hist = pd.read_csv(CSV_HISTORICO, encoding="utf-8-sig", dtype={"CVEGEO": str})
    hist = pd.concat([hist, tabla_csv], ignore_index=True)
    hist = hist.drop_duplicates(subset=["timestamp", "CVEGEO"], keep="last")
else:
    hist = tabla_csv
hist.to_csv(CSV_HISTORICO, index=False, encoding="utf-8-sig")

# Bitácora de calidad: permite ver en el tiempo cuándo la interpolación es confiable
diag = pd.DataFrame(diagnostico)
if not diag.empty:
    if CSV_DIAGNOSTICO.exists():
        diag = pd.concat([pd.read_csv(CSV_DIAGNOSTICO), diag], ignore_index=True)
        diag = diag.drop_duplicates(subset=["timestamp", "variable"], keep="last")
    diag.to_csv(CSV_DIAGNOSTICO, index=False, encoding="utf-8-sig")

print("\nListo. GeoJSON generados en:", BASE_DIR)