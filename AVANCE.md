# AVANCE - 19 Junio 2026

## Commits realizados

### 1. `c41acb8` — Rediseño visual del visualizador

- **Colores**: #2EFFD5 para vehículos contados (pasaron zona), #0F1667 para no contados
- **Paleta de zonas**: 20 colores pastel distintos para identificar cada zona
- **Opacidad dinámica**: 60% sin zonas, 80% con 1 zona, 100% con 2+ zonas
- **Labels de zona**: texto centrado dentro del bounding box, mismo color que el borde, bold (thickness=3)
- **Stats en esquina inferior izquierda**: posición fija en `(10, h - 60)`
- **Optimización ROI**: blending solo sobre el rect del bounding box en lugar de frame completo, recuperando rendimiento (~6.5 FPS vs 2.4 FPS con overlay full-frame)

### 2. `3da1b70` — Pipeline asíncrono con QueuePipe y 4 hilos

- **`domain/pipes.py`**: PipeMapper protocol, IdentityMapper, DataPipe protocol
- **`adapters/pipes/queue_pipe.py`**: QueuePipe thread-safe con input/output mappers
- **`application/pipelined_pipeline.py`**: PipelinedPipeline con 4 hilos independientes:
  - Hilo 1 (Reader): lee frames + preprocesa
  - Hilo 2 (Detector): inferencia (soporta batch)
  - Hilo 3 (Tracker + Visualizer): tracking, zone check, escritura a cola
  - Hilo 4 (Writer): escritura de video a disco
- **`PipelineMode`** enum (SEQUENTIAL, PIPELINED) en `domain/enums.py`
- **`PipelineConfig.pipeline_mode`** con soporte `from_dict`/`to_dict`
- Parámetros `pipeline_mode` y `queue_maxsize` en facade `process_video`
- Tests: 11 tests en `test_pipes.py`, 10 tests en `test_pipelined_pipeline.py`
- **Benchmark** (RTX 3060, batch=12): Sequential 46.2s / 6.5 FPS → Pipelined 42.3s / 7.1 FPS (~9% ganancia)

### 3. `3efa279` — Profiling por paso con Profiler + CSV

- **`utils/profiler.py`**: Clase Profiler thread-safe con soporte de bloqueo por hilo
- **`ProfileRecord`**: dataclass con campos `video`, `frame`, `step`, `duration_ms`, `batch`
- Integración en `Pipeline.process_video` (secuencial): timing en preprocess, detect, track, zone_check, visualize (incluye flush batch)
- Integración en `PipelinedPipeline.process_video` (pipelined): timing en los 4 workers vía closures y paso de parámetros a `_flush_raw`
- Parámetros `profile: bool` y `profile_path: str | None` en facade `process_video` y CLI `--profile`
- 11 tests en `test_profiler.py` (single/multi-thread, CSV export, edge cases)
- 136 tests unitarios totales pasando

## Benchmark comparativo (CPU, YOLOv8s, reduce=2, 297 frames)

| Métrica | Sequential | Pipelined | Ganancia |
|---|---|---|---|
| Tiempo total | 60.1s | 48.0s | **-20%** |
| FPS promedio | 4.9 | 6.2 | **+26%** |
| detect avg | 137.1ms | 154.6ms | (+12% overhead cola) |
| track avg | 49.2ms | 51.0ms | |
| zone_check avg | 0.9ms | 1.5ms | |
| visualize avg | 6.9ms | 9.0ms | |

Resultados idénticos en detecciones (15) y conteo por zona en ambos modos.

## Pendientes (próximos pasos)

### 1. Cambiar Profiler a timestamps absolutos

Actualmente el profiler usa `duration_ms` (float calculado como `(time.perf_counter() - t) * 1000`). Se necesita cambiar a:

- `start_time: float` — timestamp absoluto de inicio del paso
- `end_time: float` — timestamp absoluto de fin del paso
- La duración se puede derivar de la diferencia, pero se conservan las marcas para análisis más flexibles (diagramas de Gantt, superposición entre hilos en modo pipelined)

Esto implica modificar `ProfileRecord`, `Profiler.record`, y todos los puntos de llamada en `pipeline.py` y `pipelined_pipeline.py`.

### 2. Generar reporte HTML con gráficos dinámicos

Actualmente el profiler solo exporta un CSV. Se requiere un método adicional que genere un reporte HTML autónomo (sin dependencias externas) con:

- **Chart.js** (vía CDN) para gráficos dinámicos:
  - Barras apiladas por frame (desglose de cada paso)
  - Línea de FPS a lo largo del video
  - Distribución de tiempos por paso (histograma o box plot)
  - Tabla resumen con estadísticas (avg, min, max, p50, p95 por paso)
- Inline CSS para diseño limpio y responsivo
- El reporte debe abrirse directamente en el navegador sin necesidad de servidor

El método podría llamarse `Profiler.save_html(path: str) -> str` y ser invocado junto con `save()`.
