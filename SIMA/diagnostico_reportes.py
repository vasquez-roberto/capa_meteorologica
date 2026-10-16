import re
import requests

URLS = [
    (
        "SURESTE",
        "https://aire.nl.gob.mx/SIMA2017reportes/ReporteDiariosimaIcars_L3.php?estacion1=SURESTE",
    ),
    (
        "SAN Pedro",
        "https://aire.nl.gob.mx/SIMA2017reportes/ReporteDiariosimaIcars_L3.php?estacion1=[SAN%20Pedro]",
    ),
    (
        "NORESTE2",
        "https://aire.nl.gob.mx/SIMA2017reportes/ReporteDiariosimaIcars_L3.php?estacion1=NORESTE2",
    ),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
}

session = requests.Session()

for nombre, url in URLS:
  print(f"\n==========================================")
  print(f" ANALIZANDO REPORTE: {nombre}")
  print(f"==========================================")

  resp = session.get(url, headers=HEADERS, timeout=12)
  html = resp.text

  # 1. Buscar todos los scripts .php, .json o .js llamados en el código
  scripts = set(re.findall(r'[\w_]+\.(?:php|json|js)', html, re.IGNORECASE))
  print(f"Archivos/Scripts referenciados:")
  for s in sorted(scripts):
    print(f" - {s}")

  # 2. Buscar URLs completas o relativas con parámetros en llamadas fetch/AJAX/iFrames
  peticiones = set(
      re.findall(r'[\w_]+\.php\?[^"\'\s>]+', html, re.IGNORECASE)
  )
  print(f"\nRutas PHP con parámetros halladas:")
  for p in sorted(peticiones):
    print(f" - {p}")

  # 3. Verificar si el texto 'Humedad', 'Temperatura' o 'Radiación' existe directamente en el HTML
  coincidencias_met = re.findall(
      r"(?:temperatura|humedad|radiaci[oó]n)[^<]*", html, re.IGNORECASE
  )
  print(f"\nCoincidencias de texto meteorológico en el HTML:")
  for m in coincidencias_met[:5]:
    print(f" - {m.strip()}")