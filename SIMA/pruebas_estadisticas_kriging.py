from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging

# -------------------------------------------------------------------------
# 1. Configuración de Rutas y Variables
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
CSV_SIMA_LECTURAS = BASE_DIR / "sima_meteo_lecturas.csv"

# Configuración directa de las variables y sus variogramas
VARIABLES_CONFIG = {
    "Temperatura_C": {"modelo": "spherical", "unidad": "°C"},
    "Humedad_pct": {"modelo": "spherical", "unidad": "%"},
    "RadiacionSolar_Wm2": {"modelo": "linear", "unidad": "W/m²"}
}

print(f"Cargando datos brutos desde {CSV_SIMA_LECTURAS}...\n")
df_sima = pd.read_csv(CSV_SIMA_LECTURAS)

# -------------------------------------------------------------------------
# 2. Procesamiento LOOCV para cada Variable (Sin Filtros)
# -------------------------------------------------------------------------
for var, config in VARIABLES_CONFIG.items():
    if var not in df_sima.columns:
        print(f" La columna {var} no existe en el CSV. Omitiendo...\n")
        continue

    df_temp = df_sima.copy()
    
    # Conversión a numérico (reemplazando solo cadenas 'ND' por NaN para evitar errores de tipo)
    df_temp[var] = pd.to_numeric(df_temp[var].replace("ND", np.nan), errors="coerce")
    
    # Solo elimina registros donde la variable o las coordenadas son estrictamente NaN
    df_clean = df_temp.dropna(subset=[var, "lat", "lon"]).reset_index(drop=True)

    lons = df_clean["lon"].values
    lats = df_clean["lat"].values
    valores_reales = df_clean[var].values
    n_puntos = len(valores_reales)

    if n_puntos < 3:
        print(f" {var}: Insuficientes estaciones ({n_puntos}). Se requieren al menos 3.\n")
        continue

    # Validación Cruzada LOOCV
    valores_predichos = []
    varianzas_kriging = []

    for i in range(n_puntos):
        train_lons = np.delete(lons, i)
        train_lats = np.delete(lats, i)
        train_vals = np.delete(valores_reales, i)
        
        target_lon = np.array([lons[i]])
        target_lat = np.array([lats[i]])
        
        try:
            OK = OrdinaryKriging(
                train_lons,
                train_lats,
                train_vals,
                variogram_model=config["modelo"],
                verbose=False,
                enable_plotting=False,
                exact_values=True,
            )
            z_est, ss_var = OK.execute("points", target_lon, target_lat)
            valores_predichos.append(z_est.data[0])
            varianzas_kriging.append(ss_var.data[0])
        except Exception:
            valores_predichos.append(np.nan)
            varianzas_kriging.append(np.nan)

    valores_predichos = np.array(valores_predichos)
    varianzas_kriging = np.array(varianzas_kriging)

    # Cálculo de métricas
    errores = valores_predichos - valores_reales
    me = np.nanmean(errores)
    rmse = np.sqrt(np.nanmean(errores**2))
    mae = np.nanmean(np.abs(errores))
    msdr = np.nanmean((errores**2) / varianzas_kriging)

    ss_res = np.nansum(errores**2)
    ss_tot = np.nansum((valores_reales - np.nanmean(valores_reales))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    # Impresión estructurada en consola
    print("="*55)
    print(f" RESULTADOS SIN FILTRO - {var} ({config['unidad']})")
    print("="*55)
    print(f"Estaciones evaluadas : {n_puntos}")
    print(f"Error Medio (ME)      : {me:8.4f}  (Ideal: ~0.0)")
    print(f"RMSE                  : {rmse:8.4f}  (Precisión en {config['unidad']})")
    print(f"MAE                   : {mae:8.4f}  (Error absoluto medio)")
    print(f"MSDR                  : {msdr:8.4f}  (Ideal: ~1.0)")
    print(f"R² (Ajuste)           : {r2:8.4f}  (Ideal: cercano a 1.0)")
    print("="*55 + "\n")

    # -------------------------------------------------------------------------
    # 3. Gráficas por Variable
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Diagnóstico de Kriging sin Filtros — {var}", fontsize=14, fontweight="bold")

    # Panel 1: Real vs Predicho
    v_all = np.concatenate([valores_reales, valores_predichos])
    min_v, max_v = np.nanmin(v_all), np.nanmax(v_all)

    axes[0].scatter(valores_reales, valores_predichos, color="navy", alpha=0.7, edgecolors="k", s=50)
    axes[0].plot([min_v, max_v], [min_v, max_v], "r--", label="Línea 1:1 (Ideal)")
    axes[0].set_title(f"Ajuste de Predicción (R² = {r2:.2f})")
    axes[0].set_xlabel(f"Valor Real ({config['unidad']})")
    axes[0].set_ylabel(f"Valor Predicho LOOCV ({config['unidad']})")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Panel 2: Histograma del Error
    axes[1].hist(errores, bins=8, color="teal", edgecolor="black", alpha=0.7)
    axes[1].axvline(0, color="red", linestyle="--", label="Sin error")
    axes[1].axvline(me, color="orange", linestyle="-.", label=f"ME ({me:.2f})")
    axes[1].set_title(f"Distribución del Error (RMSE = {rmse:.2f})")
    axes[1].set_xlabel(f"Error: Predicho - Real ({config['unidad']})")
    axes[1].set_ylabel("Cantidad de Estaciones")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    # Panel 3: Incertidumbre / Varianza
    axes[2].scatter(varianzas_kriging, errores**2, color="purple", alpha=0.7, edgecolors="k", s=50)
    max_var_plot = max(np.nanmax(varianzas_kriging), np.nanmax(errores**2))
    axes[2].plot([0, max_var_plot], [0, max_var_plot], "g--", label="Línea MSDR = 1.0")
    axes[2].set_title(f"Evaluación Varianza (MSDR = {msdr:.2f})")
    axes[2].set_xlabel("Varianza Calculada por Kriging")
    axes[2].set_ylabel("Error Cuadrático Observado (Error²)")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    plt.show()