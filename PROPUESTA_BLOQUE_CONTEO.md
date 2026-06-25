# Propuesta: Bloque de Agrupación y Conteo por Flujos (`CountAggregator`)

## 1. Visión General

Se diseñará un nuevo bloque opcional (`CountAggregator`) que se ejecuta **inmediatamente después** de finalizar el procesamiento de todos los videos de una carpeta/subcarpeta. Su función es leer los archivos `_detections.csv` generados, aplicar una **agrupación jerárquica multinivel** guiada por la sección `count:` del YAML de configuración, y producir un CSV consolidado con **3 niveles de encabezado**:

```
intervalo  | norte                          | sur                      | este           | oeste
           | 1A             | 4              | 6                        |                |
           | car|moto|bus|tr|car|moto|bus|tr | car|moto|bus|tr| ...     | car|moto|...  | ...
00:00-00:14| 12 | 5  | 3 |0 | 8 | 2  | 1 |0 | 6  | 2   | 0  |0        | 3  | 1  |      | ...
00:15-00:29| 15 | 7  | 1 |2 | 10| 4  | 2 |1 | ...                      | ...           | ...
```

Donde:
- **Fila 1**: Acceso (norte, sur, este, oeste)
- **Fila 2**: Nombre del flujo (1A, 4, 6, etc.)
- **Fila 3**: Tipo de vehículo (car, motorcycle, bus, truck)
- **Columna 1**: Intervalo de tiempo (00:00-00:14, 00:15-00:29, ... 23:45-23:59)
- **Celdas**: Conteo de vehículos que coincidieron con ese flujo en ese intervalo

---

## 2. Arquitectura General

El diseño sigue los mismos patrones del código existente: **Domain** (entidades), **Application** (casos de uso), **Output** (formatos), **CLI** (comandos).

```
afora_vc/
├── domain/
│   └── counting/                       # Nuevo sub-dominio de conteo
│       ├── __init__.py
│       ├── entities.py                 # Entidades del dominio de conteo
│       ├── enums.py                    # Enums específicos (Access, etc.)
│       └── services.py                 # Servicios de agrupación
├── application/
│   ├── batch_config.py                 # Se extiende para parsear count:
│   ├── facade.py                       # Se agrega hook post-procesamiento
│   └── count_pipeline.py              # Pipeline de agrupaciones (nuevo)
├── output/
│   └── count_exporter.py              # Exportador CSV con Strategy (nuevo)
├── cli/
│   └── main.py                         # Se agrega subcomando "group"
```

### 2.1. Pipeline de Agrupación Jerárquica (Chain of Responsibility)

Cada nivel de agrupación es un **eslabón independiente** en una cadena. Cada eslabón recibe el resultado del anterior, lo transforma, y lo pasa al siguiente.

```
detections.csv
    ↓
┌─────────────────────────────────────────────────────────┐
│ Level 1: AbsoluteTimeConverter                           │
│   timestamp_formatted → tiempo absoluto                  │
│   vehicle_id → {video_name}_{vehicle_id}                 │
│   (usa timestamp y nombre del archivo de video origen)   │
└─────────────────────────────────────────────────────────┘
    ↓  List[AbsoluteZoneEntry]  (IDs únicos a través de videos)
┌─────────────────────────────────────────────────┐
│ Level 2: VehiclePathBuilder                      │
│   Agrupa entries por vehicle_id,                 │
│   construye path ordenado por tiempo             │
│   → [(vehicle_id, vehicle_type, [zona, ...])]    │
└─────────────────────────────────────────────────┘
    ↓  List[VehiclePath]
┌─────────────────────────────────────────────────┐
│ Level 3: FlowMatcher                             │
│   Empareja VehiclePath con los flujos            │
│   definidos en count.groups[].zones              │
│   → asigna flow_id + access + vehicle_type       │
│   → todo VehiclePath queda etiquetado            │
│     (flow_name o "UNKNOWN")                      │
└─────────────────────────────────────────────────┘
    ↓  List[VehiclePath] (con flow asignado)
┌──────────────────────────────────────────────────────────┐
│ Level 4: InterventionFileExporter                         │
│   Crea detections.csv con todos los VehiclePath           │
│   columnas: vehicle_id, vehicle_type, zones,              │
│            hora_inicial, hora_final, path_asignado        │
│   Paths sin match → "UNKNOWN"                             │
└──────────────────────────────────────────────────────────┘
    ↓  detections.csv (archivo de interventoría)
    ↓  List[MatchedFlow] (solo los que sí hicieron match)
┌─────────────────────────────────────────────────┐
│ Level 5: IntervalGrouper                         │
│   Agrupa por intervalo (ej: 15 min)              │
│   usando tiempo absoluto                         │
│   → {(access, flow_name, vehicle_type, interval): count}    │
└─────────────────────────────────────────────────┘
    ↓  Dict[(access, flow_name, vehicle_type, interval), int]
┌─────────────────────────────────────────────────┐
│ Level 6: CountExporter (Strategy Pattern)        │
│   Toma datos agrupados + config de columnas      │
│   → genera CSV con 3 filas de encabezado         │
└─────────────────────────────────────────────────┘
    ↓  aggregated_counts.csv
```

---

## 3. Nuevas Entidades de Dominio (`domain/counting/entities.py`)

```python
@dataclass
class AbsoluteZoneEntry:
    """ZoneEntry con tiempo absoluto derivado del nombre del video."""
    vehicle_id: str         # {video_name}_{id_original}_{vehicle_type}
    vehicle_type: str
    zone: str
    date: str               # Fecha del video (del directorio o config)
    absolute_time: str      # HH:MM:SS en tiempo absoluto
    video_relative_seconds: float  # Segundos desde inicio del video

@dataclass
class VehiclePath:
    """Trayectoria completa de un vehículo a través de zonas.
    
    El vehicle_id incluye el tipo porque la detección puede saltar
    de clase por error y un mismo ID numérico aparecer con distintos
    tipos. El composite key {video}_{id}_{type} asegura que cada
    combinación se trackee independientemente.
    """
    vehicle_id: str         # {video_name}_{id_original}_{vehicle_type}
    vehicle_type: str
    zones: list[str]        # Secuencia ordenada de zonas [A, B, C]
    first_seen: float       # Timestamp absoluto (segundos del día)
    last_seen: float        # Timestamp absoluto (segundos del día)
    flow_name: str | None = None  # Asignado por FlowMatcher, None = UNKNOWN

@dataclass
class CountGroupConfig:
    """Configuración de un grupo de conteo del YAML."""
    name: str
    include: dict[int, str] | Literal["all"]
    exclude: dict[int, str] | Literal["none"]
    access: str             # norte, sur, este, oeste
    zone_paths: list[list[str]]  # [[a, b, c, d], [a, b, c], ...]

@dataclass 
class CountConfig:
    """Configuración completa de la sección count: del YAML."""
    interval_minutes: int
    groups: list[CountGroupConfig]

@dataclass
class MatchedFlow:
    """Un vehículo emparejado con un flujo específico."""
    vehicle_id: int
    vehicle_type: str
    flow_name: str          # Nombre del grupo (ej: "1a")
    access: str             # Acceso (norte, sur, ...)
    absolute_seconds: float # Timestamp absoluto en segundos del día
    interval_key: str       # Ej: "00:00-00:14"

@dataclass
class AggregatedCell:
    """Una celda en la tabla final de conteo."""
    access: str
    flow_name: str
    vehicle_type: str
    interval: str           # "HH:MM-HH:MM"
    count: int
```

---

## 4. Servicios de Agrupación (`domain/counting/services.py`)

Cada servicio implementa un caso de uso **independiente y testeable**:

### 4.1. `AbsoluteTimeConverter`
- Lee los archivos `_detections.csv` de una carpeta
- Extrae el timestamp del nombre del video (patrón `_HH.MM.SS`)
- Convierte `timestamp_formatted` (MM:SS) a tiempo absoluto
- **Modifica el `vehicle_id`** para que sea único entre videos y tipos: `{video_name}_{id_original}_{vehicle_type}` (ej: `roosvelt_2026-06-01_15.40.13_10001_car`)
- Esto evita ambigüedades cuando un mismo ID numérico aparece con distintos tipos por errores de detección (class jumping)
- Retorna `list[AbsoluteZoneEntry]`

### 4.2. `VehiclePathBuilder`
- Agrupa `AbsoluteZoneEntry` por `vehicle_id`
- Ordena por tiempo dentro de cada grupo
- Construye `VehiclePath` con la secuencia de zonas en orden
- Retorna `list[VehiclePath]`

### 4.3. `FlowMatcher`
- Recibe `list[VehiclePath]` y `list[CountGroupConfig]`
- Para cada `VehiclePath`, determina si el path coincide con algún `zone_paths`
- Comprueba que el `vehicle_type` esté en `include` y no en `exclude`
- Asigna `flow_name` en cada `VehiclePath` (o `None` si no hay match)
- Retorna `list[VehiclePath]` con `flow_name` poblado

### 4.4. `InterventionFileExporter`
- Recibe `list[VehiclePath]` (cada uno con `flow_name` ya asignado por FlowMatcher)
- Filtra los que sí hicieron match → `list[MatchedFlow]`
- Genera archivo `detections.csv` en la carpeta procesada con las columnas:
  - `vehicle_id` — ID único prefijado
  - `vehicle_type` — tipo de vehículo
  - `zones` — lista de zonas que recorrió en orden temporal (`[A, B, C]`)
  - `hora_inicial` — hora absoluta del ingreso a la primera zona (`HH:MM:SS`)
  - `hora_final` — hora absoluta del ingreso a la última zona (`HH:MM:SS`)
  - `path_asignado` — nombre del flujo con el que matcheó, o `"UNKNOWN"` si no matcheó ninguno
- Retorna `list[MatchedFlow]` para el siguiente paso

### 4.5. `IntervalGrouper`
- Recibe `list[MatchedFlow]` y `interval_minutes`
- Calcula la clave de intervalo para cada `MatchedFlow` según su `first_seen`
- Agrupa por `(access, flow_name, vehicle_type, interval)`
- Retorna `dict[tuple[str, str, str, str], int]` (access, flow_name, vehicle_type, interval → count)

---

## 5. Pipeline de Agrupación (`application/count_pipeline.py`)

```python
class GroupPipeline:
    """Pipeline que encadena las agrupaciones jerárquicamente."""

    def __init__(self, config: CountConfig, vehicle_classes: dict[int, str] | None = None):
        self._vehicle_classes = vehicle_classes or _default_coco_classes()
        self._steps: list[GroupStep] = [
            AbsoluteTimeConverter(),
            VehiclePathBuilder(),
            FlowMatcher(config, self._vehicle_classes),
            InterventionFileExporter(),
            IntervalGrouper(config.interval_minutes),
        ]

    def execute(self, folder_path: str) -> dict[tuple[str, str, str, str], int]:
        data = None
        for step in self._steps:
            data = step.process(data, folder_path)
        return data

    def export(self, data: dict[tuple[str, str, str, str], int], exporter: CountExporterStrategy, output_path: str):
        exporter.write(data, output_path)
```

Cada `GroupStep` sigue el protocolo:

```python
class GroupStep(Protocol):
    def process(self, data: Any, context: StepContext) -> Any: ...
```

---

## 6. Output con Strategy Pattern (`output/count_exporter.py`)

```python
class CountExporterStrategy(Protocol):
    """Strategy para el formato de exportación."""
    def write(self, data: dict[tuple[str, str, str, str], int], config: CountConfig, path: str) -> None: ...

class DefaultFlowExporter:
    """ 
    Formato por defecto: CSV con 3 filas de encabezado.
    
    Fila 1: Acceso (norte, sur, oeste, este) — celdas combinadas
    Fila 2: Nombre del flujo (1A, 4, ...) — celdas combinadas
    Fila 3: Tipo de vehículo (car, motorcycle, bus, truck)
    Columna fija: intervalo (00:00-00:14, ..., 23:45-23:59)
    """
    
    ACCESS_ORDER = ["norte", "sur", "oeste", "este"]
    
    def write(self, data: dict[tuple[str, str, str, str], int], config: CountConfig, path: str) -> None:
        # 1. Construye todos los intervalos de 00:00 a 23:59
        # 2. Para cada acceso en ACCESS_ORDER, busca sus flujos en config.groups
        # 3. Para cada flujo, itera sus tipos de vehículo (include keys)
        # 4. Genera 3 filas de encabezado + N filas de datos
        # 5. Escribe CSV

class AlternativeExporter:
    """Formato alternativo (ej: columnas agrupadas por flujo directamente)."""
    def write(self, data: dict[tuple[str, str, str, str], int], config: CountConfig, path: str) -> None: ...
```

---

## 7. Integración con el Procesamiento por Lotes (`application/facade.py`)

Actualmente `process_directory` itera todos los videos de forma plana. Se modifica para:

1. Agrupar los videos pendientes por **subdirectorio** (`os.path.dirname(rel_path)`)
2. Procesar los videos uno por uno
3. **Después de cada video**, verificar si todos los videos de su subdirectorio ya fueron procesados
4. Si **no** se completó el subdirectorio → el pipeline general continúa
5. Si **sí** se completó → se dispara la tarea de conteo en un **nuevo hilo** para no bloquear el procesamiento del resto de videos

```python
import threading
from collections import defaultdict

def process_directory(...):
    ...
    from collections import defaultdict
    dir_groups: dict[str, list] = defaultdict(list)
    for entry in pending:
        dir_key = os.path.dirname(entry["rel_path"]) or "__root__"
        dir_groups[dir_key].append(entry)
    
    completed_counts: dict[str, int] = defaultdict(int)
    count_threads: list[threading.Thread] = []
    
    for dir_key, entries in dir_groups.items():
        for entry in entries:
            result = process_video(...)
            completed_counts[dir_key] += 1
            
            # ¿Todos los videos de este subdirectorio ya se procesaron?
            if count_config and dir_key != "__root__" and completed_counts[dir_key] == len(entries):
                out_subdir = os.path.join(output_base, dir_key)
                t = threading.Thread(
                    target=_run_count_pipeline,
                    args=(count_config, out_subdir, dir_key, vehicle_classes),
                    daemon=True,
                )
                t.start()
                count_threads.append(t)
    
    # Esperar a que terminen todos los hilos de conteo
    for t in count_threads:
        t.join()
    ...

def _run_count_pipeline(count_config, out_subdir, dir_key, vehicle_classes):
    """Ejecuta el pipeline de conteo en un hilo separado."""
    group_pipeline = GroupPipeline(count_config, vehicle_classes)
    data = group_pipeline.execute(out_subdir)
    exporter = DefaultFlowExporter()
    output_path = os.path.join(out_subdir, f"{dir_key.replace(os.sep, '_')}_aggregated_counts.csv")
    exporter.write(data, output_path)
```

---

### 7.1. Resolución de Clases de Vehículo

Las clases de vehículo (`vehicle_type`) se resuelven desde el archivo de configuración del modelo (`model-config` YAML), el mismo que define `vehicle_classes`, `default_threshold` y `class_thresholds`.

```
Ejemplo de model-config YAML:
  vehicle_classes:
    2: car
    3: motorcycle
    5: bus
    7: truck
  default_threshold: 0.25
  class_thresholds:
    2: 0.3
    3: 0.2
```

**Reglas de resolución:**
1. Si el `model-config` YAML tiene `vehicle_classes` → se usan esas
2. Si no, se usan las clases por defecto del detector backend (COCO: 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck)
3. Si en el `count.groups[].include` del batch YAML se definen clases por ID numérico (ej: `2: car`), se resuelven contra el map de `vehicle_classes` — si el ID está en el map se usa su nombre textual; si no, se omite la entrada

**Flujo de resolución:**
```
batch_config.yaml              model-config.yaml (opcional)
  └─ count: groups:              └─ vehicle_classes:
       include:                     2: car
         2: car                      3: motorcycle
         3: motorcycle                
         5: bus                  VehicleClassesResolver
         7: truck                    ↓
                                   CountGroupConfig con
                                   vehicle_types resueltos
                                   (solo los que existen en ambos)
```

El `VehicleClassesResolver` se pasa como dependencia al `GroupPipeline` y es inyectado desde `facade.py`, permitiendo que funcione tanto en el flujo batch como en el comando standalone `afora group`.

---

## 8. Comando CLI `afora group`

Nuevo subcomando para ejecutar la agrupación de forma **independiente** sobre carpetas ya procesadas, **sin verificar que todos los videos estén completos**. Simplemente toma los archivos `*_detections.csv` que encuentre en la carpeta y ejecuta el pipeline de agrupación.

```
afora group --folder ./output/2026-06-01 --config count.yaml
afora group --config config.yaml   # usa input/output del batch config
```

**Decisión de diseño:** El CLI `group` es un comando de intervención/reprocesamiento. No asume estado del pipeline batch. Si el usuario lo ejecuta, es porque ya sabe que los datos están listos o quiere forzar la agregación. La verificación de "todos los videos procesados" es responsabilidad exclusiva del hook batch en `facade.py`.

Implementación:

```python
# En cli/main.py
g = subparsers.add_parser("group", help="Agrupar detecciones de una carpeta procesada")
g.add_argument("--folder", "-f", help="Carpeta con detections.csv (opcional, usa input/output del config)")
g.add_argument("--config", "-c", required=True, help="YAML con config count:")
g.add_argument("--model-config", help="YAML con configuración del modelo (vehicle_classes)")
```

---

## 9. Archivos a Crear/Modificar

| Archivo | Acción | Descripción |
|---------|--------|-------------|
| `afora_vc/domain/counting/__init__.py` | Crear | Init del sub-dominio |
| `afora_vc/domain/counting/entities.py` | Crear | Entidades de dominio |
| `afora_vc/domain/counting/enums.py` | Crear | Enums (Access) |
| `afora_vc/domain/counting/services.py` | Crear | 5 servicios de agrupación (incl. InterventionFileExporter) |
| `afora_vc/application/batch_config.py` | Modificar | Agregar `CountConfig` field a `BatchConfig` |
| `afora_vc/application/count_pipeline.py` | Crear | Pipeline de agrupación |
| `afora_vc/application/facade.py` | Modificar | Hook post-procesamiento por subdirectorio |
| `afora_vc/output/count_exporter.py` | Crear | Exportador CSV con Strategy |
| `afora_vc/cli/main.py` | Modificar | Agregar subcomando `group` |
| `tests/test_count_pipeline.py` | Crear | Tests del pipeline completo |
| `tests/test_count_services.py` | Crear | Tests unitarios por servicio |
| `tests/test_count_exporter.py` | Crear | Tests del exportador |

---

## 10. Plan de Implementación (Orden Sugerido)

1. **Domain entities** (`entities.py`, `enums.py`) — tipos base
2. **CountConfig parsing** en `batch_config.py` — parsear sección `count:` del YAML
3. **AbsoluteTimeConverter** — servicio + tests con datos reales de `test-afora-vc`
4. **VehiclePathBuilder** — servicio + tests
5. **FlowMatcher** — servicio + tests
6. **IntervalGrouper** — servicio + tests
7. **CountExporter** con Strategy — output + tests
8. **GroupPipeline** — pipeline que ensambla todo
9. **Integración en facade.py** — hook post-procesamiento
10. **CLI `group`** — comando standalone
11. **VehicleClassesResolver** — servicio de resolución de clases + tests
12. **Integración asíncrona en facade.py** — hilos y contadores por subdirectorio
13. **Prueba end-to-end** con datos de `test-afora-vc`

---

## 11. Consideraciones Técnicas

### Formato de nombre de video esperado
Los videos siguen el patrón:
```
{nombre}_{YYYY-MM-DD}_{HH}.{MM}.{SS}.mp4
```
Ejemplo: `roosvelt_2026-06-01_15.40.13.mp4`
→ El tiempo absoluto se calcula como: `base = parse("15:40:13") + offset(timestamp_formatted)`

### Estructura del CSV final (3 niveles de encabezado)

```
intervalo  | norte                          | sur                      | este           | oeste
           | 1A             | 4              | 6                        |                |
           | car|moto|bus|tr|car|moto|bus|tr | car|moto|bus|tr| ...     | car|moto|...  | ...
00:00-00:14| 12 | 5  | 3 |0 | 8 | 2  | 1 |0 | 6  | 2   | 0  |0        | 3  | 1  |      | ...
```

- **Fila 1 (Acceso)**: norte, sur, oeste, este (en ese orden)
- **Fila 2 (Flujo)**: los nombres de `count.groups[].name` agrupados por su `acces`
- **Fila 3 (Tipo)**: los vehicle_type definidos en `include` de cada flujo
- **Columna fija**: intervalo (00:00-00:14, 00:15-00:29, ..., 23:45-23:59)
- **Valores**: conteo de vehículos que cumplen acceso + flujo + tipo en ese intervalo
- Las filas cubren **todo el día** (00:00 a 23:59), incluso si no hay datos (se llena con 0)

### Manejo de paths no coincidentes
Los vehículos cuyo path de zonas no coincida con ningún flujo configurado se descartan del conteo (no se asigna a ningún grupo). Esto es correcto: solo se contabilizan los flujos definidos explícitamente en el YAML.

### Include/Exclude de tipos
- `include: all` → todos los vehicle_types son válidos
- `include: {2: car, 3: motorcycle}` → solo esos tipos
- `exclude: none` → no se excluye ninguno
- `exclude: {1: bicycle}` → se excluye bicycle

### Composición del vehicle_id
El ID final incluye tres componentes separados por `_`:
```
{video_name}_{track_id_original}_{vehicle_type}
```
Ej: `roosvelt_2026-06-01_15.40.13_10001_car`

Esto asegura unicidad incluso cuando:
- Dos videos distintos tienen IDs de track overlapping
- Un mismo track_id cambia de clase por error de detección (class jumping)

### Ejecución asíncrona en el proceso batch
- El pipeline de conteo se dispara en un `threading.Thread` **daemon** para no bloquear el procesamiento de videos restantes
- Se usa un contador por subdirectorio (`completed_counts[dir_key]`) para detectar cuándo se completó el último video
- El `main thread` hace `join()` de todos los hilos de conteo al final de `process_directory`
- Esto permite que la agregación corra en paralelo con el procesamiento de otros subdirectorios

### Resolución de clases de vehículo
Las clases se resuelven en este orden de precedencia:
1. `vehicle_classes` del `model-config` YAML (si se proporcionó)
2. Clases COCO por defecto: `{1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}`
3. En el `FlowMatcher`, los `include` del YAML se resuelven contra el map de clases para obtener el `vehicle_type` textual
4. Si un `class_id` del `include` no existe en el map de clases, se omite (log warning)

Se introduce `VehicleClassesResolver` como dependencia inyectable en `GroupPipeline`:
- Desde `facade.py`: se construye con las clases del `model-config` o defaults
- Desde CLI `afora group`: se construye con defaults (o se provee `--model-config`)

### Intervalos
- El `interval` se define en minutos en el YAML
- Ej: `interval: 15` → intervalos de 00:00-00:14, 00:15-00:29, ..., 23:45-23:59
- El `timekey` de un `MatchedFlow` se determina como: `floor(absolute_seconds / (interval * 60)) * (interval * 60)`
