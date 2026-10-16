# Módulo de Monitoreo e Interpolación Geoestadística: Red SIMA (ZMM)

Este módulo automatiza la captura y el procesamiento geoestadístico de las variables meteorológicas oficiales del **Sistema Integral de Monitoreo Ambiental (SIMA)** de Gobierno del Estado de Nuevo León, aplicando **Kriging Ordinario** sobre la estructura territorial de **AGEB (INEGI)**.

---

## ¿Qué hace este módulo?

1. **Monitoreo Oficial**: Extrae datos en tiempo real de las 15 estaciones oficiales de la red SIMA para las variables meteorológicas de Temperatura Exterior (`TOUT`), Humedad Relativa (`RH`) y Radiación Solar (`SR`).
2. **Corrección Geométrica**: Reproyecta temporalmente la capa de AGEBs a UTM Zona 14N (`EPSG:32614`) para calcular los centroides en metros exactos antes de pasarlos al espacio geográfico `EPSG:4326`, evitando distorsiones de distancia.
3. **Modelado Geoestadístico (Kriging Ordinario)**: Modela las superficies continuas mediante `pykrige`, calculando por cada AGEB tanto la estimación del valor esperado como la varianza del error de estimación ($\sigma^2_k$) para evaluar la incertidumbre espacial.
4. **Validación Metodológica**: Incluye un script de evaluación cruzada Leave-One-Out (LOOCV) que permite medir matemáticamente el desempeño ($MAE$, $RMSE$, $R^2$) del modelo Kriging sobre la red.
5. **Exportación SIG**: Genera GeoJSONs optimizados para visualización y simbología por incertidumbre en **ArcGIS Online**.

---

## ¿Cómo lo hacemos?

### 1. Extracción Oficial (`sima_meteo.py`)
* **Parsing de Endpoints**: Se conecta a los endpoints validados de SIMA (`api_conc_correcto.php` y `api_meteo_correcto.php`).
* **Estandarización de Variables**: Elimina caracteres invisibles en blanco y mapea estrictamente las columnas `TOUT` (Temperatura °C), `RH` (Humedad %) y `SR` (Radiación $W/m^2$).
* **Salidas Generadas**:
  * `sima_meteo_lecturas.csv`: Lectura actual por estación oficial.
  * `sima_meteo_historico.csv`: Registro histórico acumulativo.

### 2. Interpolación por Kriging Ordinario (`interpolacion_meteo_sima_ageb.py`)
* **Ajuste de Variogramas**:
  * **Temperatura y Humedad**: Aplica modelo de semivariograma Esférico (`spherical`).
  * **Radiación Solar**: Aplica modelo de semivariograma Lineal (`linear`).
* **Generación de Indicadores**: Asigna a cada polígono AGEB el valor estimado y su varianza asociada (`Var_Temp`, `Var_Hum`, `Var_Rad`).
* **Archivos Exportados**:
  * `sima_Temperatura_AGEB.geojson`, `sima_Humedad_AGEB.geojson`, `sima_Radiacion_AGEB.geojson`: Capas vectoriales para ArcGIS Online.
  * `sima_interpolacion_lecturas.csv` e `sima_interpolacion_historico.csv`: Registros alfa-numéricos por AGEB.

### 3. Validación Cruzada LOOCV (`validar_confiabilidad_kriging.py`)
* **Evaluación Dejar-Uno-Fuera**: Retira iterativamente una estación, predice su valor con las $N-1$ restantes y calcula métricas globales de confiabilidad:
  * **MAE (Error Absoluto Medio)**: $6.49\text{ °C}$
  * **RMSE (Error Cuadrático Medio)**: $7.90\text{ °C}$
  * **$R^2$ Score**: $-0.165$ (Demuestra alta incertidumbre al usar únicamente 15 puntos para toda la ZMM).
  * **Varianza Promedio ($\sigma^2$)**: $57.52$

---

## ¿Cómo correrlo?

### 1. Requisitos Previos

Asegúrate de contar con las librerías necesarias instaladas:

```bash
pip install pandas geopandas numpy pykrige scikit-learn shapely
