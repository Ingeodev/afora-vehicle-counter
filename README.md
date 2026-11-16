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
| `--codec` | Codec del video: `avc1`/`h264`, `hevc`/`h265`, `mp4v` | `avc1` |
| `--crf` | Calidad H.264/H.265 (0-51, más alto = más chico) | `30` |
| `--preset` | Preset x264 (`ultrafast`, `veryfast`, `medium`, ...) | `veryfast` |
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
| `-v, --verbose` | Logs `INFO` a stdout | `False` |
| `--log-level` | Nivel de logging (`DEBUG`, `INFO`, `WARNING`, ...) | `WARNING` |

Ver [Logging y progreso](#logging-y-progreso-tty-colab-ci) para el detalle de
cómo se comporta en Colab o CI.

### Grabaciones DVR de Dahua (`.dav`)

La librería lee las grabaciones **`.dav` de forma nativa**, sin convertir a MP4.
No hace falta instalar FFmpeg aparte: `opencv-python-headless` incluye FFmpeg
con el demuxer `dhav` (FFmpeg >= 4.2), que es lo que permite abrir el formato.

```bash
# Un archivo .dav suelto
afora process "grabacion.dav" --zones zonas.json --output ./resultados

# Una carpeta completa con dias de grabacion
afora scan "/ruta/CAM 2" --output ./resultados --zones zonas.json
```

Cómo se interpreta la estructura del DVR:

```
ruta/AAAA-MM-DD/001/dav/HH/HH.MM.SS-HH.MM.SS[...].dav
```

- La fecha sale de la carpeta `AAAA-MM-DD` y la hora del nombre del archivo.
- Los segmentos **contiguos de un mismo día se unen en un solo video lógico**:
  el tracker no se reinicia cada pocos minutos y los vehículos que cruzan el
  borde entre archivos no se cuentan dos veces.
- La salida se organiza por fecha:
  `resultados/AAAA-MM-DD/dav_AAAA-MM-DD_HH.MM.SS_*.csv`. Ese nombre es el que
  consume el paso de conteo.
- En `.dav`, `--date` y `--base-time` se ignoran: la fecha y hora reales salen
  de la ruta.
- Si hay huecos reales (grabación por movimiento), cada bloque contiguo se
  procesa como un video independiente.

> Requisito: OpenCV con FFmpeg >= 4.2. Los wheels oficiales lo cumplen.
> Verificar con `python -c "import cv2; print(cv2.getBuildInformation())"`.

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

#### Referencia de campos del YAML

**General**

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `input` | `str` | — (requerido) | Carpeta raíz con los videos |
| `output` | `str` | `./output` | Carpeta base de salida |
| `recursive` | `bool` | `true` | Escanear subcarpetas |
| `mask` | `str` | `null` | Máscara global (PNG) |
| `zones` | `str` | `null` | Zonas globales (JSON) |
| `quiet` | `bool` | `false` | Silenciar barras y paneles |

**Detección y tracking**

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `backend` | `str` | `yolo` | `yolo`, `dfine` o `rtdetr` |
| `model` | `str` | `models/yolov8s.pt` | Ruta al modelo |
| `device` | `str` | auto | `cpu`, `cuda`, `mps` |
| `strategy` | `str` | `box` | `box` o `seg` |
| `tracker` | `str` | `botsort` | `botsort` o `bytetrack` |
| `tracker_config` | `str` | `null` | YAML de configuración del tracker |
| `model_config` | `str` | `null` | YAML con thresholds/clases por clase |
| `pipeline_mode` | `str` | `sequential` | `sequential` o `pipelined` |
| `batch_size` | `int` | `1` | Tamaño de batch (>1 activa batch mode) |
| `queue_maxsize` | `int` | `4` | Cola máxima en modo pipelined |
| `reduce` | `int` | `1` | Factor de reducción (2 = media resolución) |
| `max_minutes` | `float` | `null` | Limitar a N minutos por video |
| `night_enhance` | `bool` | `false` | Mejora nocturna |
| `deblurring` | `bool` | `false` | Deblurring agresivo |

**Video de salida**

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `no_video` | `bool` | `false` | No escribir video procesado |
| `output_name` | `str` | `null` | Nombre del video de salida |
| `codec` | `str` | `avc1` | `avc1`/`h264`, `hevc`/`h265`, `mp4v` (legacy) |
| `crf` | `int` | `30` | Calidad 0-51 (más alto = más chico) |
| `preset` | `str` | `veryfast` | Preset x264: `ultrafast`, `veryfast`, `medium`, ... |

**Fechas y profiling**

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `base_time` | `str` | `00:00` | Hora base (`HH:MM`); ignorado en `.dav` |
| `date` | `str` | `2025-01-01` | Fecha (`YYYY-MM-DD`); ignorado en `.dav` |
| `profile` | `bool` | `false` | Habilitar profiling por paso |
| `profile_path` | `str` | `null` | Ruta del CSV de profiling |
| `audit` | `bool` | `false` | Auditar detecciones por etapa |
| `overrides` | `list` | `[]` | Ajustes por video (ver más abajo) |
| `count` | `dict` | `null` | Conteo agregado (ver "Conteo y gráficos") |

#### Video de salida

El video anotado se codifica con H.264 (`libx264`) usando PyAV, que trae
FFmpeg con `libx264`/`libx265` en sus wheels. OpenCV **no** incluye `libx264`
y por eso antes `avc1` caía a `mp4v`, generando archivos enormes.

```yaml
codec: avc1       # avc1|h264, hevc|h265 o mp4v (legacy OpenCV)
crf: 30           # 0-51. Más alto = más chico y más rápido de escribir
preset: veryfast  # ultrafast, superfast, veryfast, medium, slow...
```

Los videos de salida son para verificación, no para archivo final. Con
`crf: 30` y `preset: veryfast` un clip de 1080p pesa ~6x menos que con `mp4v`
(medido: 361 KB vs 2.3 MB en un clip de 3 s). Si necesitás más detalle en los
overlays bajá el `crf` (por ejemplo 23).

`mp4v` sigue disponible para compatibilidad, pero usa el backend legacy de
OpenCV e ignora `crf` y `preset` (además de pesar mucho más).

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

Si todos los videos ya estaban procesados, `run` termina con exit code `0` sin
generar archivos nuevos. Para ignorar `progress.csv` en una corrida puntual:

```bash
afora run config.yaml --no-resume
```

#### Overrides por video

Se pueden sobreescribir parámetros para videos específicos:

```yaml
overrides:
  - path: 2026-06-01/video1.mp4
    mask: C:\mask_especial.png
    reduce: 1
    batch_size: 8
    codec: hevc
    crf: 35
```

Los overrides aceptan los mismos campos de video que el YAML global:
`mask`, `zones`, `reduce`, `batch_size`, `pipeline_mode`, `model`, `backend`,
`tracker`, `device`, `strategy`, `max_minutes`, `night_enhance`,
`deblurring`, `no_video`, `codec`, `crf`, `preset`, `output_name`,
`model_config`, `tracker_config`, `base_time`, `date`, `queue_maxsize`,
`profile`, `profile_path` y `audit`. Los valores del override pisan al global
solo para ese video.

### Logging y progreso (TTY, Colab, CI)

Por defecto el CLI no imprime logs: solo los paneles de resumen y el progreso.
Para ver los logs del paquete (backend elegido, encoder del video, peso final,
errores) usá:

```bash
afora run config.yaml -v            # INFO a stdout
afora run config.yaml --log-level DEBUG
```

| Flag | Descripción | Default |
|---|---|---|
| `-v, --verbose` | Logs a nivel `INFO` | desactivado |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `WARNING` |

Los logs se escriben a **stdout** (no a stderr) con `force=True`, así se ven en
Jupyter/Colab, en CI y en cualquier runner que capture stdout. Se pueden pasar
antes o después del subcomando (`afora -v run ...` o `afora run ... -v`).

El progreso se adapta al entorno:

- **Terminal real (TTY)**: barras de `tqdm` con refresco en vivo.
- **Sin TTY (Colab, notebooks, CI, salida redirigida)**: sin barras ni `\r`; se
  emite una línea cada 30 s con el porcentaje y una línea compacta por video:

  ```
  Procesando video_processed.mp4: 45% (5000/11000)
  [1/4] video1.mp4 -> OK (11000 frames, 320.4s)
  ```

Si redirigís la salida a un archivo y querés las barras igual, forzalas con:

```bash
AFORA_PROGRESS=1 afora run config.yaml
```

`quiet: true` en el YAML (o `--quiet`) sigue silenciando ambos formatos.


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
    codec="avc1",       # avc1|h264, hevc|h265, mp4v
    crf=30,             # 0-51, más alto = archivo más chico
    preset="veryfast",  # preset x264
)

print(f"Detecciones: {result.total_detections}")
print(f"Tiempo: {result.processing_time_minutes:.1f} min")
print(f"Zonas: {result.zone_counts}")
```

Los parámetros `codec`, `crf` y `preset` también están disponibles en
`process_directory` y en `PipelineConfig.video`:

```python
from afora_vc.application.config import PipelineConfig

config = PipelineConfig()
config.video.codec = "hevc"
config.video.crf = 32
config.video.preset = "ultrafast"
config.output.save_video = True
```

`video` también acepta una grabación DVR de Dahua: una ruta `.dav` o una lista
de segmentos `.dav` que se procesan como un solo video continuo.

```python
result = process_video(
    video=[
        "CAM 2/2026-11-04/001/dav/10/10.00.00-10.05.00[R][0@0][0][-18000].dav",
        "CAM 2/2026-11-04/001/dav/10/10.05.00-10.10.00[R][0@0][0][-18000].dav",
    ],
    zones="zonas.json",
    output="./output",
)
```

### Procesar directorio

```python
from afora_vc.application.facade import process_directory

results = process_directory(
    root_path="./videos/",          # tambien acepta arboles de .dav
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
│   ├── video/            # Lectura/escritura de video (incluye .dav de Dahua)
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
