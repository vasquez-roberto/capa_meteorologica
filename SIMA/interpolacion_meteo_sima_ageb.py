"""
=============================================================================
INTERPOLACIÓN SIMA (15 ESTACIONES) -> AGEBs DEL INEGI   (v3, con QC integrado)
=============================================================================
Variables: PM10, PM2.5, Temperatura, Humedad Relativa, Radiación Solar.

QUÉ ESTABA PASANDO (por qué salían -650 °C y RMSE de 2677)
----------------------------------------------------------
El SIMA marca un sensor sin dato con el valor centinela -9999, y muchos
sensores caídos reportan 0.0 exacto (temperatura 0.0, humedad 0.0, radiación
0.0). Eso NO son mediciones. Cuando entraron al kriging:
  - un solo -9999 arruina el variograma -> RMSE de miles
  - los 0.0 de sensores caídos fingen "zonas de 0 °C / 0 % de humedad"
  - la radiación venía en kW/m2 (0.0-0.7) mezclada con W/m2
El método de interpolación NO era el problema; los DATOS de entrada sí.

QUÉ HACE ESTE SCRIPT (en orden)
-------------------------------
1. CONTROL DE CALIDAD por variable (cada sensor se evalúa por separado: una
   estación puede tener PM bueno y temperatura caída):
     a) -9999 y otros centinelas            -> faltante
     b) Temperatura = 0.0 / Humedad = 0.0   -> faltante (sensor sin dato)
     c) Fuera de rango físico               -> faltante
     d) Radiación = 0.0 de día (la red mide > 50 W/m2) -> faltante
     e) Copias exactas entre estaciones     -> faltante (el servidor devolvió
        la misma estación dos veces; no se sabe cuál es la real)
     f) Outlier espacial (T, HR, Rad): muy lejos de la mediana de la red en
        esa hora (p. ej. 1.4 °C en una estación cuando las demás marcan 21 °C)
   Radiación: se unifica y se convierte kW/m2 -> W/m2 si hace falta.
   PM NO se filtra por outlier espacial: un foco local de partículas puede ser real.

2. Para cada variable se usan SOLO las estaciones válidas y se compite entre
   métodos con VALIDACIÓN CRUZADA leave-one-out (quita una estación, la
   predice con las demás, compara):
     - Kriging (exponential / spherical / linear)   <- preferido si empata
     - Thin-plate spline (varios suavizados)
     - IDW (varias potencias)
     - Media de la red (línea base)
   Si varios métodos quedan a menos de TOL_EMPATE del mejor error, gana el
   Kriging exponencial (el variograma que mejor te funcionó). Si ninguno mejora
   a la media por lo menos MIN_SKILL, se usa la media y se avisa.

3. Se estima SOLO donde hay estaciones válidas alrededor: por variable se usa el
   casco convexo de SUS estaciones válidas + BUFFER_KM. Fuera de ahí el AGEB
   queda en null (no se extrapola; mejor un hueco que un dato inventado).

COLUMNAS DE CADA GEOJSON (ArcGIS)
---------------------------------
  <variable>   valor estimado           Error_CV     error típico (RMSE de CV)
  Dist_Est_km  distancia a la estación válida más cercana
  Calidad      Alta / Media / Baja (según skill del método y distancia)
  Metodo, Skill, N_Estaciones, Fecha

Uso:
    python sima_interpolacion.py            # normal
    python sima_interpolacion.py --ranking  # tabla de métodos por variable
=============================================================================
"""
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist

try:
    from pykrige.ok import OrdinaryKriging
    HAY_PYKRIGE = True
except ImportError:
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

CRS_METRICO = 32614      # UTM 14N (metros). Las distancias se calculan en km.
BUFFER_KM = 2.0          # margen alrededor de las estaciones válidas
MIN_ESTACIONES = 5       # mínimo de estaciones válidas para interpolar
MARGEN_RANGO = 0.15      # se permite salir 15% del rango observado
MIN_SKILL = 0.10         # el método debe mejorar >=10% a la media; si no, media
TOL_EMPATE = 0.03        # métodos a <3% del mejor RMSE se consideran empatados
DIST_BAJA_KM = 6.0       # más lejos que esto, la calidad baja un nivel

VARIABLES = {
    "PM10": dict(col="PM10_ugm3", salida="sima_PM10_AGEB.geojson",
                 rango=(1.0, 1000.0), lim_min=0.0, lim_max=None),
    "PM25": dict(col="PM25_ugm3", salida="sima_PM25_AGEB.geojson",
                 rango=(0.5, 500.0), lim_min=0.0, lim_max=None),
    "Temperatura": dict(col="Temperatura_C", salida="sima_Temperatura_AGEB.geojson",
                        rango=(-5.0, 50.0), lim_min=None, lim_max=None),
    "Humedad": dict(col="Humedad_pct", salida="sima_Humedad_AGEB.geojson",
                    rango=(1.0, 100.0), lim_min=0.0, lim_max=100.0),
    "Radiacion": dict(col="RadiacionSolar_Wm2", salida="sima_Radiacion_AGEB.geojson",
                      rango=(0.0, 1400.0), lim_min=0.0, lim_max=None),
}
COLS = [v["col"] for v in VARIABLES.values()]

# Outlier espacial: umbral = 3*MAD, acotado entre piso y techo (unidades de la variable)
OUTLIER = {
    "Temperatura_C": dict(piso=4.0, techo=8.0),
    "Humedad_pct": dict(piso=15.0, techo=30.0),
    "RadiacionSolar_Wm2": dict(piso=200.0, techo=350.0),
}
RAD_DIA_WM2 = 50.0       # si la mediana de la red supera esto, es de día

# =============================================================================
# 2. LECTURA Y CONTROL DE CALIDAD
# =============================================================================
MARCAS_GIT = ("<<<<<<<", "=======", ">>>>>>>")


def cargar_lecturas(path):
    """Lee el CSV (tolera marcas de conflicto de git) y devuelve el ÚLTIMO timestamp."""
    lineas = Path(path).read_text(encoding="utf-8-sig").splitlines()
    texto = "\n".join(l for l in lineas if not l.startswith(MARCAS_GIT))
    df = pd.read_csv(io.StringIO(texto), dtype=str, keep_default_na=False,
                     on_bad_lines="skip")
    df = df[df["timestamp"].str.match(r"^\d{4}-\d{2}-\d{2}")]
    for c in COLS + ["RadiacionSolar_kWhm2", "lat", "lon"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")        # 'ND' -> NaN
    ult = df["timestamp"].max()
    df = df[df["timestamp"] == ult].drop_duplicates("Clave", keep="last")
    return df.reset_index(drop=True), ult


def control_calidad(df):
    """Anula (NaN) las lecturas inválidas. Devuelve (df_limpio, bitácora)."""
    df = df.copy()
    log = []

    def anular(mask, col, motivo):
        mask = np.asarray(mask) & df[col].notna().values
        for i in np.where(mask)[0]:
            log.append((col, df["Estacion"].iloc[i], df[col].iloc[i], motivo))
        df.loc[mask, col] = np.nan

    # a) centinelas (-9999) en cualquier columna de datos
    for c in COLS + ["RadiacionSolar_kWhm2"]:
        anular(df[c] <= -999, c, "centinela -9999")

    # Radiación: hay dos columnas posibles; se unifican y se pasa a W/m2.
    rad = df["RadiacionSolar_kWhm2"].combine_first(df["RadiacionSolar_Wm2"])
    pos = rad[rad > 0]
    if len(pos) and pos.max() <= 5:          # valores 0-5 => son kW/m2
        rad = rad * 1000.0
    df["RadiacionSolar_Wm2"] = rad
    df = df.drop(columns=["RadiacionSolar_kWhm2"])

    # b) ceros = sensor sin dato (en Monterrey no se mide 0.0 exacto)
    anular(df["Temperatura_C"] == 0, "Temperatura_C", "cero = sin dato")
    anular(df["Humedad_pct"] == 0, "Humedad_pct", "cero = sin dato")

    # c) rangos físicos
    for v in VARIABLES.values():
        lo, hi = v["rango"]
        anular((df[v["col"]] < lo) | (df[v["col"]] > hi), v["col"], "fuera de rango")

    # d) radiación 0.0 de día
    med_rad = df["RadiacionSolar_Wm2"].median()
    if pd.notna(med_rad) and med_rad > RAD_DIA_WM2:
        anular(df["RadiacionSolar_Wm2"] == 0, "RadiacionSolar_Wm2",
               "0.0 en horas de sol (sensor caído)")

    # e) copias exactas entre estaciones
    def copias(cols, min_nz, solo_decimales=False):
        sub = df[cols].dropna()
        sub = sub[(sub != 0).sum(axis=1) >= min_nz]
        if solo_decimales:
            sub = sub[((sub % 1) != 0).any(axis=1)]
        dup = sub.duplicated(keep=False)
        return df.index.isin(sub.index[dup])

    m = copias(["PM10_ugm3", "PM25_ugm3"], 2, solo_decimales=True)
    for c in ("PM10_ugm3", "PM25_ugm3"):
        anular(m, c, "copia idéntica en otra estación")
    m = copias(["Temperatura_C", "Humedad_pct", "RadiacionSolar_Wm2"], 2)
    for c in ("Temperatura_C", "Humedad_pct", "RadiacionSolar_Wm2"):
        anular(m, c, "copia idéntica en otra estación")

    # f) outlier espacial (variables suaves)
    for c, r in OUTLIER.items():
        x = df[c]
        if x.count() < 5:
            continue
        med = x.median()
        desv = (x - med).abs()
        mad = desv.median() * 1.4826
        umbral = float(np.clip(3 * mad, r["piso"], r["techo"]))
        anular((desv > umbral).values, c, f"outlier espacial (mediana red {med:.1f})")

    return df, log


# =============================================================================
# 3. MÉTODOS DE INTERPOLACIÓN  f(xy_tr, z_tr, xy_pr, **params) -> valores
#    xy en km, z en unidades de la variable.
# =============================================================================
def metodo_media(xy_tr, z_tr, xy_pr):
    """Línea base: todos reciben el promedio. Si algo no la supera, no sirve."""
    return np.full(len(xy_pr), z_tr.mean())


def metodo_idw(xy_tr, z_tr, xy_pr, power):
    """IDW: promedio ponderado por 1/distancia^power. Nunca sale del rango."""
    d = np.maximum(cdist(xy_pr, xy_tr), 1e-6)
    w = 1.0 / d ** power
    return (w * z_tr).sum(axis=1) / w.sum(axis=1)


def metodo_tps(xy_tr, z_tr, xy_pr, smoothing):
    """Thin-plate spline con tendencia lineal; 'smoothing' relaja el ajuste."""
    mu, sd = z_tr.mean(), z_tr.std()
    if sd == 0:
        return np.full(len(xy_pr), mu)
    f = RBFInterpolator(xy_tr, (z_tr - mu) / sd, kernel="thin_plate_spline",
                        smoothing=smoothing, degree=1)
    return f(xy_pr) * sd + mu


def metodo_kriging(xy_tr, z_tr, xy_pr, modelo):
    """Kriging ordinario; variograma ajustado solo con las estaciones válidas.
    exact_values=False permite 'nugget' (ruido de medición) => superficie más
    suave y estable que forzar el paso exacto por cada estación."""
    ok = OrdinaryKriging(xy_tr[:, 0], xy_tr[:, 1], z_tr, variogram_model=modelo,
                         nlags=6, weight=True, exact_values=False,
                         verbose=False, enable_plotting=False)
    z, _ = ok.execute("points", xy_pr[:, 0], xy_pr[:, 1])
    return np.asarray(z)


def construir_candidatos():
    c = [("Media", metodo_media, {})]
    for p in (1.0, 1.5, 2.0, 3.0):
        c.append((f"IDW p={p}", metodo_idw, {"power": p}))
    for s in (0.0, 0.2, 1.0, 5.0):
        c.append((f"TPS suav={s}", metodo_tps, {"smoothing": s}))
    if HAY_PYKRIGE:
        for m in ("exponential", "spherical", "linear"):
            c.append((f"Kriging {m}", metodo_kriging, {"modelo": m}))
    return c


PRIORIDAD = ["Kriging exponential", "Kriging spherical", "Kriging linear",
             "TPS", "IDW", "Media"]


# =============================================================================
# 4. VALIDACIÓN CRUZADA Y SELECCIÓN
# =============================================================================
def acotar(pred, z_tr, lim_min, lim_max):
    """No salir más de MARGEN_RANGO del rango observado; respetar límites físicos."""
    rango = z_tr.max() - z_tr.min()
    lo, hi = z_tr.min() - MARGEN_RANGO * rango, z_tr.max() + MARGEN_RANGO * rango
    if lim_min is not None:
        lo = max(lo, lim_min)
    if lim_max is not None:
        hi = min(hi, lim_max)
    return np.clip(pred, lo, hi)


def predecir(func, params, xy_tr, z_tr, xy_pr, lim_min, lim_max):
    return acotar(func(xy_tr, z_tr, xy_pr, **params), z_tr, lim_min, lim_max)


def loocv(func, params, xy, z, lim_min, lim_max):
    n = len(z)
    pred = np.full(n, np.nan)
    for i in range(n):
        m = np.arange(n) != i
        try:
            pred[i] = predecir(func, params, xy[m], z[m], xy[i:i + 1],
                               lim_min, lim_max)[0]
        except Exception:
            return None
    return pred if np.all(np.isfinite(pred)) else None


def ranking_metodos(xy, z, lim_min, lim_max):
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
    rm = float(rk.loc[rk["metodo"] == "Media", "rmse"].iloc[0])
    rk["skill"] = 1.0 - rk["rmse"] / rm if rm > 0 else 0.0
    return rk


def elegir_metodo(rk):
    """Empate (<TOL_EMPATE del mejor RMSE) -> prioridad por PRIORIDAD (kriging exp. primero).
    Si el elegido no supera a la media por MIN_SKILL -> Media."""
    cerca = rk[rk["rmse"] <= rk["rmse"].iloc[0] * (1 + TOL_EMPATE)].copy()
    cerca["p"] = cerca["metodo"].map(
        lambda n: next((i for i, p in enumerate(PRIORIDAD) if n.startswith(p)), 99))
    mejor = cerca.sort_values(["p", "rmse"]).iloc[0]
    if mejor["metodo"] != "Media" and mejor["skill"] < MIN_SKILL:
        return rk[rk["metodo"] == "Media"].iloc[0], True
    return mejor, False


def calidad(skill, dist_km):
    """Alta / Media / Baja según el skill del método y la distancia a estación."""
    base = 2 if skill >= 0.30 else (1 if skill >= MIN_SKILL else 0)
    pts = np.clip(base - (dist_km > DIST_BAJA_KM).astype(int), 0, 2)
    return np.array(["Baja", "Media", "Alta"])[pts]


# =============================================================================
# 5. INTERPOLAR UNA VARIABLE
# =============================================================================
def interpolar_variable(nombre, cfg, z_all, est_xy, destino_xy):
    from shapely import contains_xy
    from shapely.geometry import MultiPoint

    ok = np.isfinite(z_all)
    n = int(ok.sum())
    print(f"\n=== {nombre}: {n} estaciones válidas de {len(z_all)} ===")
    if n < MIN_ESTACIONES:
        print(f"  Se necesitan >= {MIN_ESTACIONES}. Se omite (no se publica mapa).")
        return None

    xy, z = est_xy[ok], z_all[ok]
    dist = cdist(destino_xy, xy).min(axis=1)
    hull = MultiPoint([tuple(p) for p in xy]).convex_hull.buffer(BUFFER_KM)
    dentro = contains_xy(hull, destino_xy[:, 0], destino_xy[:, 1])

    if np.isclose(z.std(), 0):
        print(f"  Valor uniforme ({z[0]:.3f}) -> constante.")
        valores, rmse, skill, metodo = np.full(len(destino_xy), z[0]), 0.0, np.nan, "Constante"
    else:
        rk = ranking_metodos(xy, z, cfg["lim_min"], cfg["lim_max"])
        if MOSTRAR_RANKING:
            print(rk[["metodo", "rmse", "mae", "sesgo", "skill"]].round(3)
                  .to_string(index=False))
        mejor, forzada_media = elegir_metodo(rk)
        if forzada_media:
            print(f"  Ningún método mejora a la media >= {MIN_SKILL:.0%}: se usa la Media.")
        metodo, rmse, skill = mejor["metodo"], float(mejor["rmse"]), float(mejor["skill"])
        print(f"  Método: {metodo} | RMSE CV: {rmse:.3f} | skill vs. media: {skill:.2f}")
        valores = predecir(mejor["func"], mejor["params"], xy, z, destino_xy,
                           cfg["lim_min"], cfg["lim_max"])

    valores = np.where(dentro, np.round(valores, 3), np.nan)
    cal = calidad(0.0 if np.isnan(skill) else skill, dist)
    print(f"  AGEBs con estimación: {int(dentro.sum())} de {len(dentro)}")
    return dict(valor=valores, rmse=round(rmse, 3), skill=skill, metodo=metodo,
                n=n, dist=np.round(dist, 2), calidad=np.where(dentro, cal, None))


# =============================================================================
# 6. PRINCIPAL
# =============================================================================
def main():
    import geopandas as gpd

    if not CSV_LECTURAS.exists():
        raise FileNotFoundError(f"No se encontró {CSV_LECTURAS}")
    if not SHP_AGEB.exists():
        raise FileNotFoundError(f"No se encontró {SHP_AGEB}")

    df, timestamp = cargar_lecturas(CSV_LECTURAS)
    print(f"Interpolación SIMA -> AGEB | lecturas de {timestamp} ({len(df)} estaciones)")

    df, log = control_calidad(df)
    if log:
        print("\nLecturas DESCARTADAS por control de calidad:")
        print(pd.DataFrame(log, columns=["variable", "estacion", "valor", "motivo"])
              .to_string(index=False))

    est = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]),
                           crs=4326).to_crs(CRS_METRICO)
    est_xy = np.column_stack([est.geometry.x, est.geometry.y]) / 1000.0

    ageb = gpd.read_file(SHP_AGEB)
    ageb = (ageb.set_crs(4326) if ageb.crs is None else ageb).to_crs(CRS_METRICO)
    col_id = "CVEGEO" if "CVEGEO" in ageb.columns else ageb.columns[0]

    # Área de estudio general: casco convexo de las 15 estaciones + buffer
    union = est.geometry.union_all() if hasattr(est.geometry, "union_all") \
        else est.geometry.unary_union
    hull = union.convex_hull.buffer(BUFFER_KM * 1000)
    puntos = ageb.geometry.representative_point()
    dentro = puntos.within(hull).values
    ageb, puntos = ageb.loc[dentro].copy(), puntos.loc[dentro]
    destino_xy = np.column_stack([puntos.x, puntos.y]) / 1000.0
    print(f"AGEBs en el área de estaciones: {len(ageb)}")

    resultados, diagnostico = {}, []
    for nombre, cfg in VARIABLES.items():
        res = interpolar_variable(nombre, cfg, df[cfg["col"]].values.astype(float),
                                  est_xy, destino_xy)
        resultados[nombre] = res
        if res:
            diagnostico.append(dict(timestamp=timestamp, variable=nombre,
                                    metodo=res["metodo"], rmse_cv=res["rmse"],
                                    skill=None if np.isnan(res["skill"]) else round(res["skill"], 3),
                                    n_estaciones=res["n"]))

    # ---- GeoJSON por variable (WGS84)
    ageb_wgs = ageb.to_crs(4326)
    for nombre, cfg in VARIABLES.items():
        res = resultados[nombre]
        if res is None:
            continue
        g = ageb_wgs[[col_id, "geometry"]].copy()
        g[cfg["col"]] = res["valor"]
        g["Error_CV"] = res["rmse"]
        g["Dist_Est_km"] = res["dist"]
        g["Calidad"] = res["calidad"]
        g["Metodo"] = res["metodo"]
        g["Skill"] = None if np.isnan(res["skill"]) else round(res["skill"], 3)
        g["N_Estaciones"] = res["n"]
        g["Fecha"] = timestamp
        g = g[[col_id, cfg["col"], "Error_CV", "Dist_Est_km", "Calidad", "Metodo",
               "Skill", "N_Estaciones", "Fecha", "geometry"]]
        g[cfg["col"]] = g[cfg["col"]].astype(object).where(g[cfg["col"]].notna(), None)
        g.to_file(BASE_DIR / cfg["salida"], driver="GeoJSON")

    # ---- CSV ancho + histórico sin duplicados
    tabla = pd.DataFrame({"timestamp": timestamp, "CVEGEO": ageb[col_id].values})
    for nombre, cfg in VARIABLES.items():
        res = resultados[nombre]
        tabla[cfg["col"]] = res["valor"] if res else np.nan
        tabla[f"Error_CV_{nombre}"] = res["rmse"] if res else np.nan
        tabla[f"Metodo_{nombre}"] = res["metodo"] if res else "ND"
    tabla_csv = tabla.fillna("ND")
    tabla_csv.to_csv(CSV_ACTUAL, index=False, encoding="utf-8-sig")

    if CSV_HISTORICO.exists():
        hist = pd.read_csv(CSV_HISTORICO, encoding="utf-8-sig", dtype={"CVEGEO": str})
        hist = pd.concat([hist, tabla_csv], ignore_index=True)
        hist = hist.drop_duplicates(subset=["timestamp", "CVEGEO"], keep="last")
    else:
        hist = tabla_csv
    hist.to_csv(CSV_HISTORICO, index=False, encoding="utf-8-sig")

    diag = pd.DataFrame(diagnostico)
    if not diag.empty:
        if CSV_DIAGNOSTICO.exists():
            diag = pd.concat([pd.read_csv(CSV_DIAGNOSTICO), diag], ignore_index=True)
            diag = diag.drop_duplicates(subset=["timestamp", "variable"], keep="last")
        diag.to_csv(CSV_DIAGNOSTICO, index=False, encoding="utf-8-sig")

    omitidas = [n for n, r in resultados.items() if r is None]
    print("\nListo. GeoJSON en:", BASE_DIR)
    if omitidas:
        print("Variables SIN mapa por falta de estaciones válidas:", ", ".join(omitidas))


if __name__ == "__main__":
    main()