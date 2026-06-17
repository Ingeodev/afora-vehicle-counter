# Afora Vehicle Counter

Librería multi-backend para detección, tracking y conteo vehicular. Soporta YOLO, D-Fine y RT-DETR como backends de detección, con BotSORT y ByteTrack para tracking.

## Instalación

```bash
pip install afora-vehicle-counter

# Con el backend que quieras
pip install "afora-vehicle-counter[yolo]"
pip install "afora-vehicle-counter[dfine]"
pip install "afora-vehicle-counter[all]"
```

## CLI

### Procesar un video

```bash
afora process video.mp4 \
  --zones zonas.json \
  --output ./resultados \
  --backend yolo \
  --model yolov8s.pt \
  --device cuda \
  --batch-size 1
```

| Flag | Descripción | Default |
|---|---|---|
| `ZONAS` | Ruta al JSON de zonas de conteo (obligatorio) | — |
| `--zones, -z` | Archivo JSON de zonas | `None` |
| `--mask, -m` | Máscara de región (PNG) | `None` |
| `--output, -o` | Carpeta de salida | `./output` |
| `--backend, -b` | Backend: `yolo`, `dfine`, `rtdetr` | `yolo` |
| `--model` | Ruta al modelo | `yolov8s.pt` |
| `--device, -d` | Dispositivo: `cpu`, `cuda`, `mps` | auto |
| `--reduce, -r` | Factor de reducción (2 = half res) | `1` |
| `--batch-size` | Tamaño de batch (>1 activa batch mode) | `1` |
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

### Procesar un directorio

```bash
afora scan ./videos/ \
  --output ./resultados \
  --recursive \
  --backend yolo
```

Busca automáticamente archivos `*mask*.png` y `*.json` de zonas en cada subdirectorio.

## API

### Uso básico

```python
from afora_vc.application.facade import process_video

result = process_video(
    video="video.mp4",
    zones="zonas.json",
    output="./output",
    backend="yolo",
    model="yolov8s.pt",
    device="cuda",
)

print(f"Detecciones: {result.total_detections}")
print(f"Tiempo: {result.processing_time_minutes:.1f} min")
print(f"Zonas: {result.zone_counts}")
```

### Configuración programática

```python
from afora_vc.application.config import PipelineConfig
from afora_vc.application.facade import _build_pipeline

config = PipelineConfig()
config.detector.backend = "yolo"
config.detector.model_path = "yolov8s.pt"
config.detector.device = "cuda"
config.detector.confidence_threshold = 0.5
config.detector.half_precision = True

config.batch.enabled = True
config.batch.batch_size = 4
config.batch.queue_maxsize = 8

config.output.save_video = True
config.output.save_csv = True
config.output.draw_zones = True

pipeline = _build_pipeline(config, config.detector.model_path, "box")
result = pipeline.process_video(video_path="video.mp4", zones_path="zonas.json")
```

### YAML de configuración

```yaml
detector:
  backend: yolo
  model_path: yolov8s.pt
  device: cuda
  confidence_threshold: 0.5
  half_precision: true
  class_thresholds:
    car: 0.6
    truck: 0.4
  vehicle_classes:
    2: car
    7: truck

tracker:
  backend: botsort
  max_age: 30
  min_hits: 3

video:
  reduce_factor: 1
  max_minutes: null
  skip_rate: 1
  output_fps: 15.0
  codec: avc1

output:
  save_video: true
  save_csv: true
  draw_zones: true
  verbose: true

batch:
  enabled: false
  batch_size: 4
  queue_maxsize: 4

preprocess:
  mode: none
```

```python
from afora_vc.application.config import PipelineConfig

config = PipelineConfig.from_yaml("config.yaml")
```

### ProcessingResult

`process_video()` devuelve un `ProcessingResult` con:

| Campo | Tipo | Descripción |
|---|---|---|
| `video_path` | `str` | Ruta al video original |
| `output_video_path` | `str\|None` | Ruta al video procesado |
| `csv_path` | `str\|None` | Ruta al CSV de detecciones |
| `total_detections` | `int` | Total de detecciones |
| `zone_counts` | `dict[str, int]` | Conteo por zona |
| `processing_time_seconds` | `float` | Tiempo total |
| `processing_time_minutes` | `float` | Tiempo en minutos (property) |
| `frames_processed` | `int` | Frames procesados |
| `detection_log` | `list[ZoneEntry]` | Log completo de detecciones |

### Procesar directorio

```python
from afora_vc.application.facade import process_directory

results = process_directory(
    root_path="./videos/",
    output_base="./resultados",
    recursive=True,
    backend="yolo",
)
```

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

## Estructura del proyecto

```
afora_vc/
├── application/       # Orquestación (pipeline, config, facade)
├── domain/            # Schemas, puertos, servicios, enums
├── adapters/          # Implementaciones concretas
│   ├── detectors/     # YOLO, D-Fine, RT-DETR
│   ├── trackers/      # BotSORT, ByteTrack
│   ├── video/         # Lectura/escritura de video
│   ├── storage/       # Sistema de archivos / S3
│   └── preprocessors/ # NoOp, Deblurring, Night Enhance
├── output/            # Visualización, exportación CSV
├── cli/               # Interfaz de línea de comandos
└── utils/             # Utilidades (device, time)
```
