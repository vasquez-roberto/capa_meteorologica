"""Reconstruye interpolacion_calidadaire_historico.csv con la corrección EPA y el control de calidad.

Lee el histórico de lecturas de cada sensor (purpleair_meteo_historico.csv) y, hora por hora,
aplica exactamente el mismo proceso que interpolacion_calidadaire_ageb.py en tiempo real:
control de calidad por sensor, corrección EPA del PM2.5 e interpolación por AGEB. También guarda
el valor sin corrección (método anterior) en columnas aparte.

Colocar en la subcarpeta 'Purple Air', junto a interpolacion_calidadaire_ageb.py, y ejecutar
UNA VEZ antes de poner en marcha la nueva versión de la interpolación:

    python "Purple Air/reconstruir_historico_interpolacion.py"                  # mismas horas que el histórico actual
    python "Purple Air/reconstruir_historico_interpolacion.py" --todas-las-horas  # todas las horas con lecturas de sensores
    python "Purple Air/reconstruir_historico_interpolacion.py" --prueba           # escribe en un archivo aparte

Antes de sobrescribir, guarda una copia del histórico actual con la fecha en el nombre.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import interpolacion_calidadaire_ageb as ia  # noqa: E402  (mismo método que en tiempo real)


def horas_a_reconstruir(historial, todas):
    disponibles = set(historial["timestamp"])
    if todas or not ia.CSV_HISTORICO.exists():
        return sorted(disponibles), set()
    actuales = set(pd.read_csv(ia.CSV_HISTORICO, encoding="utf-8-sig", usecols=["timestamp"])["timestamp"]
                   .astype(str).str.strip())
    return sorted(actuales & disponibles), actuales - disponibles


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--todas-las-horas", action="store_true",
                        help="reconstruir todas las horas del histórico de sensores, no solo las del histórico actual")
    parser.add_argument("--prueba", action="store_true",
                        help="escribir en interpolacion_calidadaire_historico_reconstruido.csv sin tocar el actual")
    args = parser.parse_args()
    inicio = time.time()

    # 1. Lecturas históricas de sensores, control de calidad y corrección (una sola vez, sobre todo el historial)
    historial = ia.leer_historial_sensores()
    if historial.empty:
        raise SystemExit(f"No hay lecturas en {ia.CSV_SENSORES_HISTORICO.name}.")
    motivo = ia.control_calidad(historial)
    corregidas = ia.corregir_lecturas(historial, motivo)
    print(f"Lecturas de sensores en rango: {len(historial):,} | pasan el control de calidad: {len(corregidas):,}")
    resumen_qc = (historial.assign(motivo=motivo).query("motivo != ''")
                  .groupby("motivo").agg(lecturas=("sensor_index", "size"), sensores=("sensor_index", "nunique")))
    if not resumen_qc.empty:
        print("Lecturas excluidas por motivo:\n" + resumen_qc.to_string())

    # 2. Horas a reconstruir
    horas, sin_lecturas = horas_a_reconstruir(historial, args.todas_las_horas)
    if sin_lecturas:
        print(f"Aviso: {len(sin_lecturas)} hora(s) del histórico actual no tienen lecturas de sensores "
              f"y no se pueden reconstruir; quedarán fuera: {sorted(sin_lecturas)[:5]}...")
    print(f"Horas a reconstruir: {len(horas)} ({horas[0]} a {horas[-1]})")

    # 3. AGEBs, censo e interpolación hora por hora
    agebs = ia.cargar_agebs(ia.ARCHIVO_SHP_COLONIAS, ia.cargar_datos_censales(ia.CSV_CENSO_INEGI))
    interpolador = ia.Interpolador(agebs)
    crudas_por_hora = dict(tuple(historial.groupby("timestamp")))
    corregidas_por_hora = dict(tuple(corregidas.groupby("timestamp")))
    vacio = historial.iloc[0:0]
    tablas, omitidas = [], []
    for k, hora in enumerate(horas, 1):
        crudas = crudas_por_hora.get(hora, vacio)
        corr = corregidas_por_hora.get(hora, vacio)
        if len(crudas) < ia.MIN_SENSORES:
            omitidas.append(hora)
            continue
        tablas.append(ia.filas_historico(agebs, hora, ia.procesar_hora(crudas, corr, interpolador)))
        if k % 10 == 0 or k == len(horas):
            print(f"  {k}/{len(horas)} horas procesadas")
    if omitidas:
        print(f"Horas omitidas por tener menos de {ia.MIN_SENSORES} sensores: {len(omitidas)}")
    tabla = pd.concat(tablas, ignore_index=True)[ia.COLUMNAS_HISTORICO]

    # 4. Escritura (respaldo + archivo temporal para no dejar el histórico a medias)
    destino = (ia.CSV_HISTORICO.with_name("interpolacion_calidadaire_historico_reconstruido.csv")
               if args.prueba else ia.CSV_HISTORICO)
    if not args.prueba and ia.CSV_HISTORICO.exists():
        respaldo = ia.CSV_HISTORICO.with_name(
            f"interpolacion_calidadaire_historico_respaldo_{time.strftime('%Y%m%d_%H%M')}.csv")
        shutil.copy2(ia.CSV_HISTORICO, respaldo)
        print(f"Respaldo del histórico anterior: {respaldo.name}")
    tmp = destino.with_suffix(".csv.tmp")
    tabla.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(destino)

    # 5. Resumen
    mediana = tabla.groupby("timestamp")[["pm25_sin_correccion", "pm25_interpolado"]].median().median()
    print(f"\nHistórico escrito: {destino.name} | {len(tabla):,} filas | {tabla['timestamp'].nunique()} horas | "
          f"{tabla['cvegeo'].nunique()} AGEBs")
    print(f"PM2.5 mediano por hora → sin corrección: {mediana['pm25_sin_correccion']:.1f} µg/m³ | "
          f"corregido: {mediana['pm25_interpolado']:.1f} µg/m³")
    print(f"Tiempo: {time.time() - inicio:.0f} s")


if __name__ == "__main__":
    main()