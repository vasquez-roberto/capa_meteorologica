# Módulo de Monitoreo e Interpolación Espacial: Red PurpleAir (ZMM)

Este módulo automatiza la captura, estandarización y modelado espacial de las lecturas meteorológicas y de calidad del aire provenientes de la red comunitaria de sensores **PurpleAir** instalados en la Zona Metropolitana de Monterrey (ZMM).

---

## ¿Qué hace este módulo?

1. **Monitoreo Continuo**: Extrae lecturas en tiempo real de material particulado ($PM_{1.0}$, $PM_{2.5}$, $PM_{10}$) y datos meteorológicos (Temperatura y Humedad Relativa) de la red de sensores PurpleAir en la ZMM.
2. **Estandarización de Datos**: Convierte automáticamente las lecturas de temperatura de grados Fahrenheit a Celsius para mantener la coherencia con la red oficial SIMA.
3. **Modelado Espacial**: Cruza la información con los polígonos **AGEB de INEGI** y la población del Censo 2020 para generar superficies continuas mediante **Triangulación de Delaunay / Interpolación Baricéntrica**.
4. **Exportación Integrada**: Genera capas vectoriales GeoJSON estructuradas para su consumo directo y representación visual en **ArcGIS Online**.

---

## ¿Cómo lo hacemos?

### 1. Extracción y Estandarización (`purpleair_meteo.py`)
* **Consulta a API**: Se conecta a la API v1 de PurpleAir (`https://api.purpleair.com/v1/sensors`) utilizando el catálogo local de sensores detectados (`sensores_detectados.csv`).
* **Conversión Térmica**: Aplica la conversión explícita:
  $$°C = \frac{F - 32}{1.8}$$
* **Generación de Salidas**: 
  * `purpleair_meteo_lecturas.csv`: Snapshot de la lectura actual por sensor.
  * `purpleair_meteo_historico.csv`: Registro histórico acumulativo.
  * `purpleair_meteo.geojson`: Capa puntual vectorial para mapeo en SIG.

### 2. Interpolación Espacial por AGEB (`interpolacion_meteo_ageb.py`)
* **Vinculación Censo-Shapefile**: Lee el shapefile de AGEBs (`2025_1_19_A.shp`) y cruza los datos del Censo INEGI 2020 (`cpv2020.csv`), estandarizando la clave `CVEGEO` a 13 dígitos.
* **Interpolación por Triangulación de Delaunay**: Construye una red de triángulos sin solapamiento basada en la ubicación de los sensores (`scipy.spatial.Delaunay`). Para cada AGEB, toma
