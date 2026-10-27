"""
=============================================================================
INTERPOLACIÓN SIMA (15 ESTACIONES) -> AGEBs DEL INEGI   (v4, sin método Media)
=============================================================================
Variables: PM10, PM2.5, Temperatura, Humedad Relativa, Radiación Solar.

POR QUÉ UNA SOLA HORA NO BASTA
------------------------------
Con 10-15 estaciones y una sola hora de lecturas, el variograma/estructura
espacial no es identificable: cualquier método vuelve a la media. Pero la red SÍ
tiene variación espacial PERSISTENTE (zonas siempre más cálidas, más húmedas,
más contaminadas). Esa señal se ve bien cuando se promedian MUCHAS horas.
Este script la extrae del histórico (sima_meteo_historico.csv).

MODELO (en cada AGEB x, hora t)
-------------------------------
    valor(x,t) = base(t) + offset(x) + residuo(x,t)

  base(t)     nivel general de la red esta hora (mediana robusta). Cambia con la
              hora del día, el clima, la nubosidad.
  offset(x)   diferencia PERSISTENTE de cada estación contra la red, medida con
              el histórico (mediana de sus anomalías en todas las horas válidas).
              Se interpola espacialmente a todos los AGEBs => mapa con variación.
              Temperatura y humedad: diferencia (°C, puntos %).
              PM y radiación: razón (se trabaja en logaritmo, nunca da negativos).
  residuo     lo que esta hora tiene de distinto al patrón habitual; se
              interpola solo si la validación cruzada demuestra que ayuda.

PASOS
-----
1. QC por hora (centinela -9999, ceros de sensores caídos, rangos, copias entre
   estaciones, radiación 0.0 de día). Radiación se pasa a W/m2.
2. Offsets por estación desde el histórico. Una estación queda FUERA del offset
   si tiene < MIN_HORAS_OFFSET horas válidas o un sesgo imposible (|offset| >
   límite físico): es un sensor descalibrado, no geografía (p. ej. una estación
   que siempre marca 9 °C menos que sus vecinas).
3. Se prueban combinaciones  offset-interpolador x residuo-interpolador:
     interpoladores: IDW, thin-plate spline, Kriging exponencial / esférico
     (el Kriging exponencial gana los empates, a <3% del mejor error).
   Se evalúan con VALIDACIÓN CRUZADA ANIDADA leave-one-out: se quita una
   estación y se predice su valor con las demás (offset y residuo incluidos).
   NO existe el método "Media": siempre hay variación espacial. La media solo se
   usa como línea base para calcular el SKILL (qué tanto mejora el modelo a un
   mapa plano) y marcar la confianza.
4. Se estima únicamente dentro del casco convexo de los 15 sensores (+BUFFER_KM).
5. Cada AGEB lleva su Calidad (Alta/Media/Baja) según el skill de la variable y
   la distancia a la estación válida más cercana. SKILL <= 0 => el modelo no
   superó a un mapa plano: se publica con Calidad "Baja", no se oculta.

Con solo ~10 estaciones válidas, el error de validación cruzada es optimista
(se eligió la mejor entre varias combinaciones). Úsalo como orden de magnitud.

COLUMNAS DE CADA GEOJSON (ArcGIS)
---------------------------------
  <variable>, Error_CV (RMSE de CV), Dist_Est_km, Calidad, Metodo, Skill,
  N_Estaciones, Fecha

También escribe sima_offsets_estaciones.csv (offset, horas y motivo de exclusión
por estación y variable) para que veas qué sensores no son confiables.

Uso:
    python sima_interpolacion.py            # normal
    python sima_interpolacion.py --ranking  # tabla de combinaciones por variable
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
CSV_HIST_EST = BASE_DIR / "sima_meteo_historico.csv"       # histórico de ESTACIONES
SHP_AGEB = ROOT_DIR / "shp" / "2025_1_19_A.shp"            # requiere .dbf y .prj

CSV_ACTUAL = BASE_DIR / "sima_interpolacion_lecturas.csv"
CSV_HISTORICO = BASE_DIR / "sima_interpolacion_historico.csv"
CSV_DIAGNOSTICO = BASE_DIR / "sima_interpolacion_diagnostico.csv"
CSV_OFFSETS = BASE_DIR / "sima_offsets_estaciones.csv"

CRS_METRICO = 32614      # UTM 14N (metros); las distancias se calculan en km
BUFFER_KM = 0.0          # 0 = hasta el límite de los sensores; sube para ampliar
MIN_ESTACIONES = 5       # mínimo de estaciones válidas para interpolar
MARGEN_RANGO = 0.15      # se permite salir 15% del rango observado
TOL_EMPATE = 0.03        # combinaciones a <3% del mejor RMSE = empate
DIST_BAJA_KM = 6.0       # más lejos que esto, la calidad baja un nivel
MIN_HORAS_OFFSET = 15    # horas válidas mínimas de una estación para su offset
MIN_ESTACIONES_BASE = 8  # estaciones válidas mínimas para calcular base(t) histórica
RAD_DIA_WM2 = 50.0       # mediana de red > esto => de día (0.0 = sensor caído)
RAD_MIN_OFFSET = 100.0   # offsets de radiación solo con sol alto
RAD_NOCHE_WM2 = 20.0     # mediana < esto => noche: radiación = 0 en todo el mapa

# transf: 'lin' (diferencias) | 'log' (razones). off_max: sesgo máximo creíble de
# una estación (en unidades transformadas). res_outlier: (piso, techo) para anular
# lecturas actuales muy distintas a base+offset (None = no filtrar; en PM un foco
# local puede ser real).
VARIABLES = {
    "PM10": dict(col="PM10_ugm3", salida="sima_PM10_AGEB.geojson",
                 rango=(1.0, 1000.0), lim_min=0.0, lim_max=None,
                 transf="log", piso_log=1.0, off_max=1.2, res_outlier=None),
    "PM25": dict(col="PM25_ugm3", salida="sima_PM25_AGEB.geojson",
                 rango=(0.5, 500.0), lim_min=0.0, lim_max=None,
                 transf="log", piso_log=0.5, off_max=1.2, res_outlier=None),
    "Temperatura": dict(col="Temperatura_C", salida="sima_Temperatura_AGEB.geojson",
                        rango=(-5.0, 50.0), lim_min=None, lim_max=None,
                        transf="lin", off_max=5.0, res_outlier=(4.0, 8.0)),
    "Humedad": dict(col="Humedad_pct", salida="sima_Humedad_AGEB.geojson",
                    rango=(1.0, 100.0), lim_min=0.0, lim_max=100.0,
                    transf="lin", off_max=15.0, res_outlier=(15.0, 30.0)),
    "Radiacion": dict(col="RadiacionSolar_Wm2", salida="sima_Radiacion_AGEB.geojson",
                      rango=(0.0, 1400.0), lim_min=0.0, lim_max=None,
                      transf="log", piso_log=1.0, off_max=1.2,
                      res_outlier=(0.8, 1.5), solo_dia=True, min_horas=8),
}
COLS = [v["col"] for v in VARIABLES.values()]


def tf(x, cfg):
    x = np.asarray(x, dtype=float)
    return np.log(np.maximum(x, cfg["piso_log"])) if cfg["transf"] == "log" else x


def tf_inv(y, cfg):
    return np.exp(y) if cfg["transf"] == "log" else y


# =============================================================================
# 2. LECTURA Y CONTROL DE CALIDAD
# =============================================================================
MARCAS_GIT = ("<<<<<<<", "=======", ">>>>>>>")


def leer_csv_tolerante(path):
    """Lee un CSV del SIMA aunque tenga marcas de conflicto de git."""
    lineas = Path(path).read_text(encoding="utf-8-sig").splitlines()
    texto = "\n".join(l for l in lineas if not l.startswith(MARCAS_GIT))
    df = pd.read_csv(io.StringIO(texto), dtype=str, keep_default_na=False,
                     on_bad_lines="skip")
    df = df[df["timestamp"].str.match(r"^\d{4}-\d{2}-\d{2}")].copy()
    for c in COLS + ["RadiacionSolar_kWhm2", "lat", "lon"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")        # 'ND' -> NaN
    return df.reset_index(drop=True)


def cargar_lecturas(path):
    df = leer_csv_tolerante(path)
    ult = df["timestamp"].max()
    df = df[df["timestamp"] == ult].drop_duplicates("Clave", keep="last")
    return df.reset_index(drop=True), ult


def cargar_historico(path):
    """Histórico de estaciones: quita duplicados exactos y filas en conflicto
    (misma hora y estación con valores distintos: no se sabe cuál es la real)."""
    df = leer_csv_tolerante(path)
    df = df.drop_duplicates()
    df = df.drop_duplicates(["timestamp", "Clave"], keep=False)
    return df.reset_index(drop=True)


def detectar_factor_rad(df):
    """El SIMA entrega la radiación en kW/m2 (0-1). Si es así, x1000 -> W/m2."""
    rad = df["RadiacionSolar_kWhm2"].combine_first(df["RadiacionSolar_Wm2"])
    pos = rad[(rad > 0) & (rad < 1e4)]
    return 1000.0 if len(pos) and pos.quantile(0.99) <= 5 else 1.0


def control_calidad(df, factor_rad):
    """Anula (NaN) lecturas inválidas de UNA hora. Devuelve (df_limpio, bitácora).
    (El filtro de outliers se hace después, contra base+offset, no contra la
    mediana: así no se borra la variación geográfica real.)"""
    df = df.copy()
    log = []

    def anular(mask, col, motivo):
        mask = np.asarray(mask) & df[col].notna().values
        for i in np.where(mask)[0]:
            log.append((col, df["Estacion"].iloc[i], df[col].iloc[i], motivo))
        df.loc[mask, col] = np.nan

    for c in COLS + ["RadiacionSolar_kWhm2"]:                 # a) centinelas
        anular(df[c] <= -999, c, "centinela -9999")

    rad = df["RadiacionSolar_kWhm2"].combine_first(df["RadiacionSolar_Wm2"])
    df["RadiacionSolar_Wm2"] = rad * factor_rad
    df = df.drop(columns=["RadiacionSolar_kWhm2"])

    anular(df["Temperatura_C"] == 0, "Temperatura_C", "cero = sin dato")   # b)
    anular(df["Humedad_pct"] == 0, "Humedad_pct", "cero = sin dato")

    for v in VARIABLES.values():                              # c) rangos
        lo, hi = v["rango"]
        anular((df[v["col"]] < lo) | (df[v["col"]] > hi), v["col"], "fuera de rango")

    med_rad = df["RadiacionSolar_Wm2"].median()               # d) 0.0 de día
    if pd.notna(med_rad) and med_rad > RAD_DIA_WM2:
        anular(df["RadiacionSolar_Wm2"] == 0, "RadiacionSolar_Wm2",
               "0.0 en horas de sol (sensor caído)")

    def copias(cols, min_nz, solo_decimales=False):           # e) copias
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
    return df, log


# =============================================================================
# 3. OFFSETS PERSISTENTES POR ESTACIÓN (desde el histórico)
# =============================================================================
def calcular_offsets(hist, factor_rad):
    """Devuelve {variable: DataFrame[Clave, Estacion, offset, n_horas, estado]}."""
    limpias = [control_calidad(g.reset_index(drop=True), factor_rad)[0]
               for _, g in hist.groupby("timestamp")]
    h = pd.concat(limpias, ignore_index=True)
    out = {}
    for nombre, cfg in VARIABLES.items():
        col = cfg["col"]
        d = h[["timestamp", "Clave", "Estacion", col]].copy()
        d["y"] = np.where(d[col].notna(), tf(d[col].fillna(1.0), cfg), np.nan)
        g = d.groupby("timestamp")
        d["base"] = g["y"].transform("median")
        d["nv"] = g["y"].transform("count")
        ok = d["y"].notna() & (d["nv"] >= MIN_ESTACIONES_BASE)
        if cfg.get("solo_dia"):
            med_orig = g[col].transform("median")
            ok &= med_orig >= RAD_MIN_OFFSET
        d = d[ok].copy()
        d["anom"] = d["y"] - d["base"]
        r = d.groupby("Clave").agg(Estacion=("Estacion", "first"),
                                   offset=("anom", "median"),
                                   n_horas=("anom", "count")).reset_index()
        min_h = cfg.get("min_horas", MIN_HORAS_OFFSET)
        estado = []
        for _, f in r.iterrows():
            # El sesgo imposible se revisa primero: -20 °C es un sensor malo
            # aunque tenga pocas horas.
            if f["n_horas"] >= 3 and abs(f["offset"]) > cfg["off_max"]:
                estado.append(f"excluida: sesgo imposible ({f['offset']:+.2f}) "
                              "= sensor descalibrado")
            elif f["n_horas"] < min_h:
                estado.append(f"excluida: solo {int(f['n_horas'])} horas válidas")
            else:
                estado.append("ok")
        r["estado"] = estado
        faltan = set(hist["Clave"]) - set(r["Clave"])       # sin ninguna hora válida
        for k in faltan:
            r.loc[len(r)] = [k, hist.loc[hist["Clave"] == k, "Estacion"].iloc[0],
                             np.nan, 0, "excluida: sin horas válidas"]
        out[nombre] = r

    # Un sensor meteorológico descalibrado en una variable no es confiable en las
    # otras (comparten estación/equipo): se excluye de temperatura, humedad y radiación.
    meteo = ("Temperatura", "Humedad", "Radiacion")
    malas = {k for n in meteo for k, e in zip(out[n]["Clave"], out[n]["estado"])
             if str(e).startswith("excluida: sesgo")}
    for n in meteo:
        r = out[n]
        m = r["Clave"].isin(malas) & ~r["estado"].str.startswith("excluida: sesgo")
        r.loc[m, "estado"] = "excluida: sensor meteorológico descalibrado (sesgo en otra variable)"
    return out


# =============================================================================
# 4. INTERPOLADORES  f(xy_tr, z_tr, xy_pr) -> valores   (xy en km)
# =============================================================================
def metodo_idw(xy_tr, z_tr, xy_pr, power):
    d = np.maximum(cdist(xy_pr, xy_tr), 1e-6)
    w = 1.0 / d ** power
    return (w * z_tr).sum(axis=1) / w.sum(axis=1)


def metodo_tps(xy_tr, z_tr, xy_pr, smoothing):
    mu, sd = z_tr.mean(), z_tr.std()
    if sd == 0:
        return np.full(len(xy_pr), mu)
    f = RBFInterpolator(xy_tr, (z_tr - mu) / sd, kernel="thin_plate_spline",
                        smoothing=smoothing, degree=1)
    return f(xy_pr) * sd + mu


def metodo_kriging(xy_tr, z_tr, xy_pr, modelo):
    """exact_values=False permite 'nugget' (ruido de medición): superficie más
    estable que forzar el paso exacto por cada estación."""
    ok = OrdinaryKriging(xy_tr[:, 0], xy_tr[:, 1], z_tr, variogram_model=modelo,
                         nlags=6, weight=True, exact_values=False,
                         verbose=False, enable_plotting=False)
    z, _ = ok.execute("points", xy_pr[:, 0], xy_pr[:, 1])
    return np.asarray(z)


def acotar(pred, z_tr, lim_min=None, lim_max=None):
    """No salir más de MARGEN_RANGO del rango observado; respetar límites físicos."""
    rango = z_tr.max() - z_tr.min()
    lo, hi = z_tr.min() - MARGEN_RANGO * rango, z_tr.max() + MARGEN_RANGO * rango
    if lim_min is not None:
        lo = max(lo, lim_min)
    if lim_max is not None:
        hi = min(hi, lim_max)
    return np.clip(pred, lo, hi)


# modos: ("cero",) | ("idw", p) | ("tps", s) | ("krig", modelo)
OFF_MODOS = {"sin offset": ("cero",), "IDW p=2": ("idw", 2.0), "IDW p=3": ("idw", 3.0),
             "TPS suav=1": ("tps", 1.0), "TPS suav=5": ("tps", 5.0)}
RES_MODOS = {"sin residuo": ("cero",), "IDW p=2": ("idw", 2.0), "TPS suav=1": ("tps", 1.0)}
if HAY_PYKRIGE:
    OFF_MODOS.update({"Kriging exponential": ("krig", "exponential"),
                      "Kriging spherical": ("krig", "spherical")})
    RES_MODOS.update({"Kriging exponential": ("krig", "exponential")})


def ejecutar(modo, xy_tr, z_tr, xy_pr):
    """Interpola z con el modo dado; None si el método falla."""
    try:
        if modo[0] == "cero":
            return np.zeros(len(xy_pr))
        if modo[0] == "idw":
            p = metodo_idw(xy_tr, z_tr, xy_pr, modo[1])
        elif modo[0] == "tps":
            p = metodo_tps(xy_tr, z_tr, xy_pr, modo[1])
        else:
            p = metodo_kriging(xy_tr, z_tr, xy_pr, modo[1])
        p = acotar(p, z_tr)
        return p if np.all(np.isfinite(p)) else None
    except Exception:
        return None


# =============================================================================
# 5. MODELO  base + offset + residuo  y validación cruzada anidada
# =============================================================================
def componentes(xy_tr, y_tr, o_tr, xy_pr):
    """Calcula, con las estaciones de entrenamiento, todas las piezas del modelo
    en los puntos xy_pr (espacio transformado)."""
    comp = {"base": {}, "off": {}, "res": {}}
    clases = {"off": (np.median(y_tr - o_tr), o_tr),     # usa offsets del histórico
              "none": (np.median(y_tr), np.zeros_like(y_tr))}
    for cls, (base, o) in clases.items():
        comp["base"][cls] = base
        r_tr = y_tr - base - o
        for nm, m in RES_MODOS.items():
            comp["res"][(cls, nm)] = ejecutar(m, xy_tr, r_tr, xy_pr)
    for nm, m in OFF_MODOS.items():
        comp["off"][nm] = ejecutar(m, xy_tr, o_tr, xy_pr)
    return comp


def combinar(comp, off_nm, res_nm):
    cls = "none" if off_nm == "sin offset" else "off"
    off, res = comp["off"][off_nm], comp["res"][(cls, res_nm)]
    if off is None or res is None:
        return None
    return comp["base"][cls] + off + res


def combos_validos(usar_off):
    nombres = [n for n in OFF_MODOS if usar_off or n == "sin offset"]
    return [(o, r) for o in nombres for r in RES_MODOS
            if not (o == "sin offset" and r == "sin residuo")]   # evita mapa plano


def validacion_cruzada(xy, v, y, o, cfg, usar_off):
    """Leave-one-out ANIDADO: el offset y el residuo de la estación retirada
    también se predicen con las demás. Errores en unidades originales."""
    combos = combos_validos(usar_off)
    n = len(v)
    pred = {c: np.full(n, np.nan) for c in combos}
    flat = np.full(n, np.nan)
    for j in range(n):
        m = np.arange(n) != j
        comp = componentes(xy[m], y[m], o[m], xy[j:j + 1])
        flat[j] = np.median(v[m])
        for c in combos:
            p = combinar(comp, *c)
            if p is not None:
                pred[c][j] = acotar(tf_inv(p, cfg), v[m], cfg["lim_min"], cfg["lim_max"])[0]
    rmse_flat = float(np.sqrt(np.mean((flat - v) ** 2)))
    filas = []
    for c, p in pred.items():
        if not np.all(np.isfinite(p)):
            continue
        e = p - v
        rmse = float(np.sqrt(np.mean(e ** 2)))
        filas.append(dict(combo=c, metodo=f"offset: {c[0]} | residuo: {c[1]}",
                          rmse=rmse, mae=float(np.mean(np.abs(e))),
                          sesgo=float(np.mean(e)),
                          skill=1.0 - rmse / rmse_flat if rmse_flat > 0 else 0.0))
    rk = pd.DataFrame(filas).sort_values("rmse").reset_index(drop=True)
    return rk, rmse_flat


def elegir(rk):
    """Empates (<TOL_EMPATE del mejor RMSE): gana la que use Kriging exponencial."""
    cerca = rk[rk["rmse"] <= rk["rmse"].iloc[0] * (1 + TOL_EMPATE)].copy()
    cerca["p"] = cerca["metodo"].map(lambda s: 0 if "Kriging exponential" in s else 1)
    return cerca.sort_values(["p", "rmse"]).iloc[0]


def calidad(skill, dist_km):
    base = 2 if skill >= 0.30 else (1 if skill >= 0.10 else 0)
    pts = np.clip(base - (dist_km > DIST_BAJA_KM).astype(int), 0, 2)
    return np.array(["Baja", "Media", "Alta"])[pts]


# =============================================================================
# 6. INTERPOLAR UNA VARIABLE
# =============================================================================
def interpolar_variable(nombre, cfg, df, est_xy, destino_xy, offsets):
    col = cfg["col"]
    v_all = df[col].values.astype(float)
    valid = np.isfinite(v_all)
    print(f"\n=== {nombre}: {int(valid.sum())} estaciones válidas de {len(v_all)} ===")

    # Radiación de noche: el sol no está, 0 W/m2 en todo el mapa (es lo físico).
    if nombre == "Radiacion" and valid.any() and np.nanmedian(v_all[valid]) < RAD_NOCHE_WM2:
        print("  Es de noche (radiación ~0): se asigna 0 W/m2 a todos los AGEBs.")
        dist = cdist(destino_xy, est_xy[valid]).min(axis=1)
        return dict(valor=np.zeros(len(destino_xy)), rmse=0.0, skill=np.nan,
                    metodo="Noche (0 W/m2)", n=int(valid.sum()),
                    dist=np.round(dist, 2), calidad=np.array(["Alta"] * len(dist)))

    usar_off = offsets is not None
    if usar_off and cfg.get("solo_dia") and np.nanmedian(v_all[valid]) < RAD_MIN_OFFSET:
        usar_off = False
        print("  Sol bajo: no se aplican offsets de radiación (solo con sol alto).")
    if usar_off:
        o_all = df["Clave"].map(offsets).values.astype(float)
        anclas = valid & np.isfinite(o_all)
        if anclas.sum() < MIN_ESTACIONES:
            usar_off = False
            print("  Pocas estaciones con offset confiable: se usa solo la hora actual.")
    if not usar_off:
        o_all = np.zeros(len(v_all))
        anclas = valid.copy()

    xy, v, o = est_xy[anclas], v_all[anclas], o_all[anclas]
    nom = df["Estacion"].values[anclas]

    # Outliers de ESTA hora contra base+offset (no contra la mediana simple)
    if cfg["res_outlier"]:
        piso, techo = cfg["res_outlier"]
        for _ in range(2):
            y = tf(v, cfg)
            res = y - np.median(y - o) - o
            mad = np.median(np.abs(res - np.median(res))) * 1.4826
            umbral = float(np.clip(3 * mad, piso, techo))
            malo = np.abs(res - np.median(res)) > umbral
            if not malo.any() or (~malo).sum() < MIN_ESTACIONES:
                break
            for k in np.where(malo)[0]:
                print(f"  Lectura descartada (outlier vs. patrón habitual): "
                      f"{nom[k]} = {v[k]:.2f}")
            xy, v, o, nom = xy[~malo], v[~malo], o[~malo], nom[~malo]
    n = len(v)
    if n < MIN_ESTACIONES:
        print(f"  Se necesitan >= {MIN_ESTACIONES} estaciones. Se omite.")
        return None
    y = tf(v, cfg)

    rk, rmse_flat = validacion_cruzada(xy, v, y, o, cfg, usar_off)
    if rk.empty:
        print("  Ningún método pudo evaluarse. Se omite.")
        return None
    if MOSTRAR_RANKING:
        print(rk[["metodo", "rmse", "mae", "sesgo", "skill"]].round(3).to_string(index=False))
    mejor = elegir(rk)
    skill = float(mejor["skill"])
    print(f"  Modelo: {mejor['metodo']}")
    print(f"  RMSE CV: {mejor['rmse']:.3f} | skill vs. mapa plano: {skill:.2f} "
          f"(RMSE plano {rmse_flat:.3f})")
    if skill <= 0:
        print("  AVISO: no supera a un mapa plano; se publica con Calidad 'Baja'.")

    comp = componentes(xy, y, o, destino_xy)
    pred_y = combinar(comp, *mejor["combo"])
    valores = acotar(tf_inv(pred_y, cfg), v, cfg["lim_min"], cfg["lim_max"])
    dist = cdist(destino_xy, xy).min(axis=1)
    print(f"  Rango estimado en AGEBs: {valores.min():.2f} a {valores.max():.2f}")
    return dict(valor=np.round(valores, 3), rmse=round(float(mejor["rmse"]), 3),
                skill=skill, metodo=mejor["metodo"], n=n,
                dist=np.round(dist, 2), calidad=calidad(skill, dist))


# =============================================================================
# 7. PRINCIPAL
# =============================================================================
def main():
    import geopandas as gpd

    for p in (CSV_LECTURAS, SHP_AGEB):
        if not p.exists():
            raise FileNotFoundError(f"No se encontró {p}")

    df, timestamp = cargar_lecturas(CSV_LECTURAS)
    print(f"Interpolación SIMA -> AGEB | lecturas de {timestamp} ({len(df)} estaciones)")

    # ---- Histórico de estaciones -> offsets persistentes
    offsets, tabla_off = {n: None for n in VARIABLES}, None
    if CSV_HIST_EST.exists():
        hist = cargar_historico(CSV_HIST_EST)
        factor_rad = detectar_factor_rad(hist)
        print(f"Histórico: {hist['timestamp'].nunique()} horas | "
              f"radiación x{factor_rad:g} -> W/m2")
        res_off = calcular_offsets(hist, factor_rad)
        filas = []
        for nombre, r in res_off.items():
            ok = r[r["estado"] == "ok"]
            offsets[nombre] = dict(zip(ok["Clave"], ok["offset"])) if len(ok) else None
            print(f"\nOffsets {nombre}:")
            print(r.sort_values("offset")[["Estacion", "offset", "n_horas", "estado"]]
                  .round(3).to_string(index=False))
            filas.append(r.assign(variable=nombre))
        pd.concat(filas).to_csv(CSV_OFFSETS, index=False, encoding="utf-8-sig")
    else:
        factor_rad = detectar_factor_rad(df)
        print(f"AVISO: no existe {CSV_HIST_EST.name}; solo se usa la hora actual "
              "(sin histórico hay mucha menos variación identificable).")

    df, log = control_calidad(df, factor_rad)
    if log:
        print("\nLecturas actuales DESCARTADAS por control de calidad:")
        print(pd.DataFrame(log, columns=["variable", "estacion", "valor", "motivo"])
              .to_string(index=False))

    est = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]),
                           crs=4326).to_crs(CRS_METRICO)
    est_xy = np.column_stack([est.geometry.x, est.geometry.y]) / 1000.0

    ageb = gpd.read_file(SHP_AGEB)
    if ageb.crs is None:
        b = ageb.total_bounds
        if abs(b[0]) <= 180 and abs(b[1]) <= 90:
            ageb = ageb.set_crs(4326)
        else:
            raise ValueError("El shapefile no trae .prj (coordenadas proyectadas). "
                             "Copia también 2025_1_19_A.prj y .dbf junto al .shp.")
    ageb = ageb.to_crs(CRS_METRICO)
    col_id = "CVEGEO" if "CVEGEO" in ageb.columns else ageb.columns[0]

    # Área de estudio: casco convexo de los sensores (+ BUFFER_KM)
    union = est.geometry.union_all() if hasattr(est.geometry, "union_all") \
        else est.geometry.unary_union
    hull = union.convex_hull.buffer(BUFFER_KM * 1000)
    puntos = ageb.geometry.representative_point()
    dentro = puntos.within(hull).values
    ageb, puntos = ageb.loc[dentro].copy(), puntos.loc[dentro]
    destino_xy = np.column_stack([puntos.x, puntos.y]) / 1000.0
    print(f"\nAGEBs dentro del polígono de los sensores: {len(ageb)}")

    resultados, diagnostico = {}, []
    for nombre, cfg in VARIABLES.items():
        res = interpolar_variable(nombre, cfg, df, est_xy, destino_xy, offsets[nombre])
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
        g.to_file(BASE_DIR / cfg["salida"], driver="GeoJSON")

    # ---- CSV ancho + histórico sin duplicados
    tabla = pd.DataFrame({"timestamp": timestamp, "CVEGEO": ageb[col_id].values})
    for nombre, cfg in VARIABLES.items():
        res = resultados[nombre]
        tabla[cfg["col"]] = res["valor"] if res else np.nan
        tabla[f"Error_CV_{nombre}"] = res["rmse"] if res else np.nan
        tabla[f"Calidad_{nombre}"] = res["calidad"] if res else "ND"
    tabla_csv = tabla.fillna("ND")
    tabla_csv.to_csv(CSV_ACTUAL, index=False, encoding="utf-8-sig")

    if CSV_HISTORICO.exists():
        hist_i = pd.read_csv(CSV_HISTORICO, encoding="utf-8-sig", dtype={"CVEGEO": str})
        hist_i = pd.concat([hist_i, tabla_csv], ignore_index=True)
        hist_i = hist_i.drop_duplicates(subset=["timestamp", "CVEGEO"], keep="last")
    else:
        hist_i = tabla_csv
    hist_i.to_csv(CSV_HISTORICO, index=False, encoding="utf-8-sig")

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