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
├── application/       # Orquestación (pipeline, config, facade, batch_config)
├── domain/            # Schemas, puertos, servicios, enums
├── adapters/          # Implementaciones concretas
│   ├── detectors/     # YOLO, D-Fine, RT-DETR
│   ├── trackers/      # BotSORT, ByteTrack
│   ├── video/         # Lectura/escritura de video
│   ├── storage/       # Sistema de archivos
│   └── preprocessors/ # NoOp, Deblurring, Night Enhance
├── output/            # Visualización, exportación CSV
├── cli/               # Interfaz de línea de comandos
└── utils/             # Utilidades (device, display, progress, geometry)
```
