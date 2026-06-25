# Afora Vehicle Counter

Librería multi-backend para detección, tracking y conteo vehicular. Soporta YOLO, D-Fine y RT-DETR como backends de detección, con BotSORT y ByteTrack para tracking.

## Instalación

### Windows (recomendado)

Clonar y ejecutar el instalador. Detecta CUDA automaticamente, crea un entorno virtual dedicado y configura las variables de entorno.

```powershell
git clone https://github.com/Ingeodev/afora-vehicle-counter.git
cd afora-vehicle-counter
powershell -ExecutionPolicy Bypass -File install.ps1
```

Lo que hace `install.ps1`:

1. Detecta la versión de CUDA via `nvcc` y selecciona el PyTorch adecuado (cu126 para CUDA 13.x)
2. Crea un `.venv` en `%PROGRAMDATA%\afora-vehicle-counter\`
3. Instala el `.whl` incluido en `dist/` (o desde source si no hay `.whl`) con extras `rtdetr,tracking,torch-cuXXX`
4. Agrega `Scripts\` del venv al `PATH` del usuario
5. Setea la variable de entorno `AFORA_VC_HOME`

Despues de instalar, reiniciar la terminal y usar:

```powershell
afora --help
```

### Manual (alternativa)

```bash
pip install "afora-vehicle-counter[rtdetr,tracking,torch-auto]"

# O con CUDA especifico:
pip install "afora-vehicle-counter[rtdetr,tracking,torch-cu126]"
```

### Extras disponibles

| Extra | Instala |
|---|---|
| `yolo` | ultralytics (backbone YOLO) |
| `dfine` | transformers (backbone D-Fine) |
| `rtdetr` | ultralytics (backbone RT-DETR) |
| `tracking` | lapx (BotSORT / ByteTrack) |
| `torch-cu126` | torch 2.6.0 + torchvision 0.21.0 |
| `torch-cu124` | torch 2.5.1 + torchvision 0.20.1 |
| `torch-cu121` | torch 2.4.0 + torchvision 0.19.0 |
| `torch-cu118` | torch 2.4.0 + torchvision 0.19.0 |
| `torch-auto` | torch>=2.0.0 (desde PyPI con CUDA incluido) |

```bash
# Todo junto
pip install "afora-vehicle-counter[rtdetr,tracking,torch-cu126]"
```

## CLI

### Procesar un video

```bash
afora process video.mp4 \
  --zones zonas.json \
  --output ./resultados \
  --backend rtdetr \
  --device cuda \
  --batch-size 16 \
  --pipeline-mode pipelined
```

| Flag | Descripción | Default |
|---|---|---|
| `--zones, -z` | Archivo JSON de zonas | `None` |
| `--mask, -m` | Máscara de región (PNG) | `None` |
| `--output, -o` | Carpeta de salida | `./output` |
| `--backend, -b` | Backend: `yolo`, `dfine`, `rtdetr` | `yolo` |
| `--model` | Ruta al modelo | `yolov8s.pt` |
| `--device, -d` | Dispositivo: `cpu`, `cuda`, `mps` | auto |
| `--reduce, -r` | Factor de reducción (2 = half res) | `1` |
| `--batch-size` | Tamaño de batch (>1 activa batch mode) | `1` |
| `--pipeline-mode` | `sequential` o `pipelined` | `sequential` |
| `--strategy` | Estrategia: `box`, `seg` | `box` |
| `--night-enhance` | Mejora nocturna | `False` |
| `--deblurring` | Deblurring agresivo | `False` |
| `--tracker` | Backend tracker: `botsort`, `bytetrack` | `botsort` |
| `--no-video` | No guardar video de salida | `False` |
| `--quiet, -q` | Modo silencioso | `False` |
| `--max-minutes` | Limitar procesamiento a N minutos | — |
| `--base-time, -t` | Hora base (HH:MM) | `00:00` |
| `--date` | Fecha del video (YYYY-MM-DD) | `2025-01-01` |
| `--model-config` | YAML con thresholds/clases por clase | — |
| `--tracker-config` | YAML de configuración del tracker | — |
| `--queue-maxsize` | Tamaño máximo de cola (modo pipelined) | `4` |
| `--profile` | Habilitar profiling por paso | `False` |
| `--profile-path` | Ruta del CSV de profiling | — |
| `--audit` | Auditar detecciones por etapa | `False` |

### Procesar lote por YAML (con reanudacion)

```bash
afora run config.yaml
```

El archivo YAML define la carpeta de entrada, salida, mascara global, zonas y parámetros de procesamiento:

```yaml
input: C:\Videos\hik\roosvelt
output: C:\output\roosvelt
recursive: true

mask: C:\mask.png
zones: C:\zones.json

backend: rtdetr
model: models/rtdetr-l.pt
pipeline_mode: pipelined
reduce: 2
batch_size: 16
```

Todos los parámetros del comando `process` están disponibles en el YAML:

```yaml
tracker_config: C:\tracker.yaml
model_config: C:\model_config.yaml
base_time: "06:00"
date: "2026-06-01"
queue_maxsize: 8
profile: true
audit: true
```

#### Conteo y gráficos

La sección `count:` habilita el pipeline de conteo agregado por subdirectorio. Al terminar de procesar todos los videos de una carpeta, genera automáticamente un CSV agregado y un gráfico.

```yaml
count:
  interval: 15           # minutos por intervalo
  graph: web             # none | image | web
  groups:
    - name: Flujo Norte
      access: norte
      include: all
      exclude: none
      zones:
        - [d, l, a]
        - [l, a]
    - name: Flujo Sur
      access: sur
      include: all
      exclude: none
      zones:
        - [c, d]
```

Los flujos se definen con:
- **`name`** — nombre del flujo (aparece en CSV y gráfico)
- **`access`** — acceso: `norte`, `sur`, `este`, `oeste`
- **`include`** / **`exclude`** — tipos de vehículo: `all`, `none`, o mapa `{class_id: nombre}`
- **`zones`** — secuencias de zonas que definen el flujo (subsequence matching)
- **`graph`** — tipo de gráfico: `none` (solo CSV), `image` (PNG con Plotly), `web` (HTML con Chart.js vía CDN)

El flag `--graph` del CLI sobreescribe el valor del YAML:

```bash
afora run config.yaml --graph image
```

#### Reanudacion automatica

El comando `run` crea/lee `progress.csv` en la misma carpeta del YAML. Cada video completado se registra con su ruta, output, tiempo en minutos y detecciones. Si se interrumpe y se vuelve a ejecutar, los videos ya procesados se saltan automaticamente.

```
input_path,output_path,processing_time_minutes,total_detections
C:\Videos\roosvelt\video1.mp4,C:\output\roosvelt,26.59,8269
```

#### Overrides por video

Se pueden sobreescribir parámetros para videos específicos:

```yaml
overrides:
  - path: 2026-06-01/video1.mp4
    mask: C:\mask_especial.png
    reduce: 1
    batch_size: 8
```

### Agrupar detecciones (post-procesamiento)

Para ejecutar solo el pipeline de conteo sobre detecciones ya existentes (sin reprocesar videos):

```bash
afora group --folder C:\output\cam1 --config config.yaml
```

| Flag | Descripción |
|---|---|
| `--folder, -f` | Carpeta con `*_detections.csv` (usa `output:` del YAML si se omite) |
| `--config, -c` | YAML con sección `count:` (requerido) |
| `--model-config` | YAML con `vehicle_classes` para resolver tipos |
| `--output, -o` | Carpeta de salida (por defecto: la misma `--folder`) |
| `--graph` | Sobreescribe `graph:` del YAML |

### Utilidades

#### Organizar videos por fecha

```bash
afora utils group-by-date C:\Videos\cam3
```

Ordena los videos en subcarpetas `YYYY-MM-DD` según el timestamp en el nombre del archivo (formato `..._YYYYMMDDHHMMSS_...`). Crea un log `_separation_log.txt` para poder revertir.

```bash
# Revertir
afora utils group-by-date C:\Videos\cam3 --rollback
```

## API

### Uso básico

```python
from afora_vc.application.facade import process_video

result = process_video(
    video="video.mp4",
    zones="zonas.json",
    output="./output",
    backend="rtdetr",
    device="cuda",
)

print(f"Detecciones: {result.total_detections}")
print(f"Tiempo: {result.processing_time_minutes:.1f} min")
print(f"Zonas: {result.zone_counts}")
```

### Procesar directorio

```python
from afora_vc.application.facade import process_directory

results = process_directory(
    root_path="./videos/",
    output_base="./resultados",
    recursive=True,
    backend="rtdetr",
)
```

### Procesar desde YAML

```python
from afora_vc.application.facade import process_from_config

results = process_from_config("config.yaml")
```

### ProcessingResult

| Campo | Tipo | Descripción |
|---|---|---|
| `video_path` | `str` | Ruta al video original |
| `output_video_path` | `str\|None` | Ruta al video procesado |
| `csv_path` | `str\|None` | Ruta al CSV de detecciones |
| `total_detections` | `int` | Total de detecciones |
| `zone_counts` | `dict[str, int]` | Conteo por zona |
| `processing_time_seconds` | `float` | Tiempo total en segundos |
| `processing_time_minutes` | `float` | Tiempo en minutos (property) |
| `frames_processed` | `int` | Frames procesados |
| `detection_log` | `list[ZoneEntry]` | Log completo de detecciones |

## Backends

| Backend | Instalación | GPU | Batch | FP16 |
|---|---|---|---|---|
| **YOLO** | `[yolo]` | ✅ | ✅ | ✅ |
| **D-Fine** | `[dfine]` | ✅ | ✅ | ✅ |
| **RT-DETR** | `[rtdetr]` | ✅ | ✅ | ✅ |

## Trackers

| Tracker | Instalación |
|---|---|
| **BotSORT** | `[tracking]` |
| **ByteTrack** | `[tracking]` |

## Pipeline modes

- **sequential** — procesa frame por frame: leer, detectar, tracker, escribir (bajo throughput, mínimo consumo de memoria)
- **pipelined** — 4 hilos concurrentes (reader, detector, tracker, writer) con colas. Mayor throughput en GPU. Recomendado para lotes con `batch_size >= 16`.

## Estructura del proyecto

```
afora_vc/
├── application/          # Orquestación (pipeline, config, facade, batch_config, count_pipeline)
├── domain/               # Schemas, puertos, servicios, enums
│   └── counting/         # Entidades y servicios del pipeline de conteo
├── adapters/             # Implementaciones concretas
│   ├── detectors/        # YOLO, D-Fine, RT-DETR
│   ├── trackers/         # BotSORT, ByteTrack
│   ├── video/            # Lectura/escritura de video
│   ├── storage/          # Sistema de archivos
│   └── preprocessors/    # NoOp, Deblurring, Night Enhance
├── output/               # Visualización, exportación CSV, gráficos (Plotly, Chart.js)
├── cli/                  # Interfaz de línea de comandos
├── utilities/            # Utilidades standalone (video_sorter)
└── utils/                # Utilidades (device, display, progress, geometry, auditor, profiler)
```

## Flujo de trabajo completo

Ejemplo práctico desde cero: organizar videos por fecha, configurar el conteo y procesar.

### 1. Organizar videos por fecha

Los videos crudos están en `C:\Videos\cam3` con nombres como `cam3_20260601143000_01.mp4`. El timestamp debe estar en el tercer campo separado por guion bajo con formato `YYYYMMDDHHMMSS`.

```powershell
afora utils group-by-date C:\Videos\cam3
```

Resultado:
```
C:\Videos\cam3\
├── 2026-06-01\
│   ├── cam3_20260601143000_01.mp4
│   └── cam3_20260601150000_02.mp4
├── 2026-06-02\
│   └── cam3_20260602120000_03.mp4
└── _separation_log.txt
```

### 2. Crear archivo de configuración

Crear `config.yaml` en el directorio de trabajo:

```yaml
input: C:\Videos\cam3
output: C:\output\cam3
recursive: true

backend: rtdetr
model: models/rtdetr-l.pt
pipeline_mode: pipelined
reduce: 2
batch_size: 12

count:
  interval: 15
  graph: web
  groups:
    - name: Flujo Norte
      access: norte
      include: all
      exclude: none
      zones:
        - [d, l, a]
        - [l, a]
    - name: Flujo Sur
      access: sur
      include: all
      exclude: none
      zones:
        - [f, i, c]
        - [i, c]
```

### 3. Ejecutar el procesamiento

```powershell
afora run config.yaml
```

El pipeline:
1. Escanea `C:\Videos\cam3` recursivamente
2. Procesa cada video con RT-DETR y BotSORT
3. Por subdirectorio (cada fecha), al terminar todos los videos:
   - Genera `detections.csv` con intervenciones (paths asignados)
   - Genera `aggregated_counts.csv` con conteo por acceso → flujo → tipo → intervalo
   - Genera `aggregated_chart.html` con gráficos de barras apiladas por flujo (Chart.js)

### 4. Resultados

```
C:\output\cam3\
├── 2026-06-01\
│   ├── cam3_20260601143000_01.mp4        # video procesado
│   ├── cam3_20260601143000_01_detections.csv  # detecciones crudas
│   ├── detections.csv                    # intervenciones (paths asignados)
│   ├── aggregated_counts.csv             # conteo agregado
│   └── aggregated_chart.html             # gráfico interactivo
├── 2026-06-02\
│   ├── cam3_20260602120000_03.mp4
│   ├── cam3_20260602120000_03_detections.csv
│   ├── detections.csv
│   ├── aggregated_counts.csv
│   └── aggregated_chart.html
└── progress.csv                          # control de reanudación
```

### 5. Ver resultados

Abrir `aggregated_chart.html` en el navegador para ver los gráficos interactivos por flujo, o leer el CSV:

```csv
access,flow_name,vehicle_type,interval,count
norte,Flujo Norte,car,06:00-06:15,12
norte,Flujo Norte,truck,06:00-06:15,3
norte,Flujo Norte,car,06:15-06:30,8
sur,Flujo Sur,bus,06:00-06:15,2
```

### Resumen del pipeline de conteo

```
detections.csv (crudas)
  → AbsoluteTimeConverter (tiempo absoluto + composite vehicle_id)
  → VehiclePathBuilder (trayectorias por vehículo)
  → FlowMatcher (asignación a flujos definidos)
  → InterventionFileExporter (detections.csv con paths)
  → IntervalGrouper (agregación por intervalos)
  → aggregated_counts.csv + aggregated_chart.{png|html}
```
