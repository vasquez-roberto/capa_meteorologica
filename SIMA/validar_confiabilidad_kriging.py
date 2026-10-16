from pathlib import Path
import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
CSV_SIMA_LECTURAS = BASE_DIR / "sima_meteo_lecturas.csv"

if not CSV_SIMA_LECTURAS.exists():
  raise FileNotFoundError(f"No se encontró {CSV_SIMA_LECTURAS}")

df = pd.read_csv(CSV_SIMA_LECTURAS)
df["Temperatura_C"] = pd.to_numeric(
    df["Temperatura_C"].replace("ND", np.nan), errors="coerce"
)
df_temp = df.dropna(subset=["Temperatura_C", "lat", "lon"]).reset_index(
    drop=True
)

lons = df_temp["lon"].values
lats = df_temp["lat"].values
vals = df_temp["Temperatura_C"].values

y_real, y_pred, varianzas = [], [], []

# Validación Cruzada Leave-One-Out (LOOCV)
for i in range(len(vals)):
  train_lons = np.delete(lons, i)
  train_lats = np.delete(lats, i)
  train_vals = np.delete(vals, i)

  test_lon = np.array([lons[i]])
  test_lat = np.array([lats[i]])

  try:
    OK = OrdinaryKriging(
        train_lons,
        train_lats,
        train_vals,
        variogram_model="spherical",
        verbose=False,
        enable_plotting=False,
    )
    z_est, ss_var = OK.execute("points", test_lon, test_lat)
    y_real.append(vals[i])
    y_pred.append(z_est.data[0])
    varianzas.append(ss_var.data[0])
  except Exception:
    continue

rmse = np.sqrt(mean_squared_error(y_real, y_pred))
mae = mean_absolute_error(y_real, y_pred)
r2 = r2_score(y_real, y_pred)
var_promedio = np.mean(varianzas)

print("--- EVALUACIÓN DE CONFIABILIDAD DEL MODELO KRIGING ---")
print(f"Estaciones evaluadas: {len(y_real)}")
print(f"MAE (Error Absoluto Medio): {mae:.2f} °C")
print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.2f} °C")
print(f"R² Score (Ajuste del Modelo): {r2:.3f}")
print(f"Varianza Promedio de Estimación: {var_promedio:.2f}")