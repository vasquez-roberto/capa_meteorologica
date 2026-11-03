# 🗺️ Capa Meteorológica — Red SIMA Monterrey (Interpolación Espacial)

Este módulo forma parte del proyecto **Capa Meteorológica** y se encarga de extraer, limpiar, procesar e **interpolar espacialmente** las lecturas meteorológicas y de contaminación de la Red de Monitoreo Ambiental (**SIMA**) de Monterrey, Nuevo León, proyectándolas sobre las **AGEBs (Áreas Geoestadísticas Básicas / Colonias)** del INEGI.

---

## 📌 Descripción General

Monitorear una zona metropolitana de más de 5 millones de habitantes con solo **15 estaciones fijas** presenta un reto de resolución espacial. Este repositorio resuelve esa limitación utilizando un **modelo de interpolación híbrido espacial-temporal**, el cual aprovecha el histórico de mediciones (`sima_meteo_historico.csv`) para capturar los microclimas y sesgos persistentes de la región.

---

## 🔬 Modelo Matemático de Interpolación

El script de interpolación (`sima_interpolacion.py`) descompone el valor de cada variable ($PM_{10}$, $PM_{2.5}$, Temperatura, Humedad Relativa y Radiación Solar) en **3 componentes independientes**:

$$\text{Valor}(x, t) = \text{Base}(t) + \text{Offset}(x) + \text{Residuo}(x, t)$$

Donde:
* **$x$**: Ubicación geográfica de la colonia/AGEB.
* **$t$**: Hora específica de la lectura.

### Componentes del Modelo

1. **$\text{Base}(t)$ (Nivel General de la Red):**
   * Corresponde a la **mediana robusta** de todas las estaciones válidas en la hora $t$. Capta eventos globales (ej. olas de calor, frentes fríos o contingencias ambientales en toda la ciudad).
2. **$\text{Offset}(x)$ (Sesgo Geográfico Persistente):**
   * Mide qué tanto se desvía históricamente una zona respecto a la media de la red (ej. García suele ser más seco y cálido que Chipinque).
   * **Variables lineales** (Temperatura, Humedad): Diferencia directa en $\text{°C}$ o $\text{\%}$.
   * **Variables logarítmicas** ($PM_{10}$, $PM_{2.5}$, Radiación): Se trabajan en espacio logarítmico para garantizar valores siempre positivos tras la transformación inversa.
3. **$\text{Residuo}(x, t)$ (Anomalía Horaria Distribuida):**
   * Es la variación local de la hora actual tras remover la base y el offset. Solo se incluye si la validación cruzada demuestra que reduce el error de predicción.

---

## ⚙️ Algoritmos y Selección Automática

Para mapear los campos continuos de **Offset** y **Residuo** entre estaciones, el sistema evalúa dinámicamente las siguientes técnicas de interpolación:

| Algoritmo | Código | Descripción |
| :--- | :--- | :--- |
| **Inverse Distance Weighting** | `idw` ($p=2, 3$) | Ponderación por distancia inversa. Robusto y rápido. |
| **Thin-Plate Splines** | `tps` | Suavizado mediante splines de placa delgada. |
| **Kriging Ordinario** | `krig` (Exponencial / Esférico) | Geoestadística basada en variogramas con tolerancia a ruido de medición (`nugget`). |

### Validación Cruzada Anidada (Leave-One-Out)
El script selecciona la combinación óptima para cada hora:
1. Retira una estación a la vez y predice su valor utilizando las $N-1$ estaciones restantes.
2. Calcula el **RMSE** (Root Mean Square Error) frente al valor real observado.
3. Elige automáticamente la combinación con menor RMSE. En empates ($<3\%$ de diferencia), prioriza **Kriging Exponencial**.

---

## 🛡️ Filtros de Calidad y Confiabilidad

* **Casco Convexo (Convex Hull):** La interpolación se limita estrictamente al polígono formado por la red de sensores (+ un buffer opcional), evitando invención de datos en zonas lejanas sin cobertura.
* **Control de Calidad Automático (QC):**
  * Descarta valores centinela (`-9999`), ceros en temperatura/humedad y lecturas de radiación en $0.0\text{ W/m}^2$ durante el día (sensores caídos).
  * Detecta estaciones descalibradas y elimina lecturas duplicadas en tiempo real.
* **Métrica de Calidad por AGEB (Alta / Media / Baja):**
  * Clasifica cada AGEB según el **Skill** del modelo ($1 - \text{RMSE} / \text{RMSE}_{\text{plano}}$) y la distancia a la estación válida más cercana ($<6\text{ km}$).

---

## 📁 Archivos y Estructura

```text
├── sima_captura.py               # Extractor automático de lecturas SIMA (manejo de sesiones/cookies)
├── sima_interpolacion.py         # Motor de interpolación híbrido y validación cruzada
├── sima_meteo_lecturas.csv       # Captura de la hora actual por estación
├── sima_meteo_historico.csv      # Base acumulativa de lecturas históricas
├── sima_PM10_AGEB.geojson        # Capa espacial de PM10 por AGEB
├── sima_PM25_AGEB.geojson        # Capa espacial de PM2.5 por AGEB
├── sima_Temperatura_AGEB.geojson # Capa espacial de Temperatura por AGEB
├── sima_Humedad_AGEB.geojson     # Capa espacial de Humedad Relativa por AGEB
├── sima_Radiacion_AGEB.geojson   # Capa espacial de Radiación Solar por AGEB
├── sima_offsets_estaciones.csv   # Diagnóstico de sesgos e historial de sensores
└── sima_interpolacion_diagnostico.csv # Bitácora de desempeño de los modelos
