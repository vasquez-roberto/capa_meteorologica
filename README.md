# Sistema de Automatización y Monitoreo Meteorológico: SIMA y PurpleAir (ZMM)

Este repositorio contiene la arquitectura automatizada para la captura, estandarización e interpolación geoestadística de datos meteorológicos y de calidad del aire en la Zona Metropolitana de Monterrey (ZMM), utilizando la red oficial del **Sistema Integral de Monitoreo Ambiental (SIMA)** y la red comunitaria de sensores **PurpleAir**.

El pipeline completo se ejecuta de forma horaria y desatendida mediante **GitHub Actions**, proyectando las estimaciones e incertidumbres sobre la estructura de polígonos **AGEB de INEGI**.

---

## Estructura del Repositorio

```text
├── .github/
│   └── workflows/
│       └── actualizar_capas.yml         # Pipeline centralizado de ejecución horaria
├── Purple Air/
│   ├── .env                             # Credenciales locales (ignorado en Git)
│   ├── sensores_detectados.csv          # Catálogo de sensores en la ZMM
│   ├── purpleair_meteo.py               # Extracción y conversión F° a C°
│   ├── interpolacion_meteo_ageb.py      # Interpolación por Triangulación de Delaunay
│   ├── purpleair_meteo_lecturas.csv     # Snapshot actual
│   ├── purpleair_meteo_historico.csv    # Registro acumulativo
│   ├── purpleair_meteo.geojson          # Puntos vectoriales
│   ├── AQ_Temperatura_AGEB.geojson      # Capa vectorial AGEB de Temperatura
│   └── AQ_Humedad_AGEB.geojson          # Capa vectorial AGEB de Humedad
├── SIMA/
│   ├── sima_meteo.py                    # Extracción oficial (TOUT, RH, SR)
│   ├── interpolacion_meteo_sima_ageb.py # Geoestadística por Kriging Ordinario
│   ├── validar_confiabilidad_kriging.py # Evaluación Leave-One-Out (LOOCV)
│   ├── sima_meteo_lecturas.csv          # Snapshot actual
│   ├── sima_meteo_historico.csv         # Registro acumulativo
│   ├── sima_interpolacion_lecturas.csv  # Resultados por AGEB actual
│   ├── sima_interpolacion_historico.csv # Histórico de interpolación por AGEB
│   ├── sima_Temperatura_AGEB.geojson    # Capa de Temperatura + Varianza
│   ├── sima_Humedad_AGEB.geojson        # Capa de Humedad + Varianza
│   └── sima_Radiacion_AGEB.geojson      # Capa de Radiación Solar + Varianza
└── shp/
    ├── 2025_1_19_A.shp                  # Polígonos de AGEB (INEGI)
    └── cpv2020.csv                      # Datos censales de población (INEGI)
