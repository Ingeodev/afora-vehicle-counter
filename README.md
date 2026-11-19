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

#### Desde el repositorio (clon local)

```bash
git clone https://github.com/Ingeodev/afora-vehicle-counter.git
cd afora-vehicle-counter
pip install -e ".[rtdetr,tracking]"

# O con CUDA especifico:
pip install -e ".[rtdetr,tracking,torch-cu126]"
```

#### Directo desde GitHub (sin clonar)

```bash
pip install "git+https://github.com/Ingeodev/afora-vehicle-counter.git#egg=afora-vehicle-counter[rtdetr,tracking]"
```

> El paquete **no esta publicado en PyPI** aun. `pip install "afora-vehicle-counter"`
> va a fallar. Usa los comandos de arriba o el instalador de Windows.

#### Google Colab

```python
!git clone https://github.com/Ingeodev/afora-vehicle-counter.git
%cd afora-vehicle-counter
!pip install -q -e ".[rtdetr,tracking]"

# Verificar la version instalada:
!afora --version
```

> **Importante:** despues de cada `git pull` o cambio de codigo, reinstalar con
> `!pip install -e ".[rtdetr,tracking]"` para que el CLI use la version actual.
> Si ves un banner con `afora 2.0.0a1` seguido de una ruta en `/usr/local/lib/...`
> que no es tu clon, el CLI esta ejecutando otra copia instalada.

#### Extras disponibles

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

La sección `count:` habilita el pipeline de conteo agregado por subdirectorio. `afora run` lo ejecuta automáticamente al terminar cada carpeta de fecha; `afora group` lo corre solo sobre detecciones ya existentes (ver [Agrupar detecciones](#agrupar-detecciones-post-procesamiento)).

```yaml
count:
  interval: 15           # minutos por intervalo
  graph: web             # none | image | web
  groups:
    - name: "1"
      access: norte
      include: all
      exclude: person
      zones:
        - [A]
        - [A, F]

    - name: "8"
      access: sur
      include: [car, motorcycle, bus, truck, bicycle]
      exclude: person
      zones:
        - [C, B, A, F]
        - [+, B, A, F]
```

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `interval` | `int` | `15` | Minutos por intervalo de agregación. Entero positivo (usá divisores de 60 para etiquetas limpias). |
| `graph` | `str` | `web` | `none`, `image` o `web`. |
| `groups` | `list` | `[]` | Flujos a contar. Cada elemento es un grupo. |

Cualquier otra clave dentro de `count:` es un error de configuración, no se ignora en silencio.

El flag `--graph` del CLI sobreescribe el valor del YAML:

```bash
afora run config.yaml --graph image
```

##### Grupos

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `name` | `str` | `""` | Nombre del flujo: es la etiqueta del CSV y del gráfico. Tiene que ser **único en toda la config**. |
| `access` | `str` | `""` | Acceso: `norte`, `sur`, `oeste` o `este`. |
| `include` | `str` / `list` / `map` | `all` | Tipos de vehículo que entran al flujo. |
| `exclude` | `str` / `list` / `map` | `none` | Tipos de vehículo que quedan afuera. |
| `zones` | `list[list[str]]` | — | Patrones de trayectoria. Requerido y no vacío. |

**`access` no es texto libre.** Tiene que ser exactamente uno de `norte`, `sur`, `oeste`, `este`, en minúsculas. Si escribís otra cosa (`Norte`, `norte `, `N`) o lo omitís, el flujo se cuenta internamente pero **no aparece** en `aggregated_counts.csv`: el exportador recorre los accesos en el orden fijo `norte → sur → oeste → este` y descarta lo que no está en esa lista. El gráfico sí lo muestra (itera los grupos del YAML), así que el CSV y el gráfico quedan inconsistentes. No hay warning.

**`name` también tiene que ser único** (no solo dentro de un acceso). El `access` de cada trayectoria se resuelve por nombre de flujo, así que con dos grupos homónimos todas las trayectorias se asignan al **último acceso declarado** y en el CSV quedan columnas duplicadas: unas con el conteo y otras en cero.

**`include` / `exclude`** aceptan todas estas formas:

```yaml
include: all                  # todos los tipos del modelo (default)
include: person               # un solo tipo
include: [car, motorcycle]    # una lista
include:
  2: car                      # mapa class_id -> nombre (compatibilidad)
exclude: none                 # ninguno (default)
exclude: person               # un solo tipo
exclude: [person, bicycle]
```

Reglas de resolución:

1. `exclude` gana siempre: si el tipo está en `exclude`, el flujo no lo cuenta aunque esté en `include`.
2. Los nombres se comparan sin distinguir mayúsculas (`Car` == `car`).
3. `all` significa todas las clases del modelo. Con `model_config` en el YAML (o `--model-config` en `afora group`) salen de `vehicle_classes`; sin él, de los defaults `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`.
4. Los nombres desconocidos se ignoran. **Ojo:** `include: [van]` contra un modelo que no tiene `van` resuelve a lista vacía, y una lista vacía significa "sin restricción" → el flujo cuenta todo. Verificá los nombres contra tu modelo.
5. El orden de la lista no importa.

##### Patrones de zonas

Cada elemento de `zones` es un patrón; un grupo puede tener todos los que necesite (se combinan con OR). El patrón es una **lista ordenada de tokens**:

```
patron := token ("," token)*
token  := <etiqueta de zona> | "_" | "+"
```

| Token | Significado | Ejemplo |
|---|---|---|
| etiqueta | Una zona concreta, tal cual figura en el JSON de zonas | `A`, `B`, `Norte` |
| `_` | Exactamente una zona, cualquiera | `[A, _, D]` |
| `+` | Una o más zonas, cualesquiera | `[A, +]` |

Reglas:

- **El match es completo.** El patrón tiene que cubrir la trayectoria entera, de la primera zona a la última. No es "contiene" ni "subsecuencia": si sobra o falta una zona, no matchea.
- La trayectoria es la secuencia de **entradas a zonas en orden cronológico**. Si un vehículo vuelve a entrar a la misma zona, la etiqueta se repite (`[A, B, A]`).
- `_` nunca matchea cero zonas: `[A, _, D]` no matchea `[A, D]`.
- `+` nunca matchea cero zonas: `[A, +]` no matchea `[A]`. Para aceptar también `[A]` hay que declarar el patrón `[A]` aparte.
- Los comodines pueden ir en cualquier posición: al inicio (`[+, D]`), al final (`[A, +]`), en el medio (`[A, +, D]`) o combinados (`[A, _, +, D]`).
- Dos `+` consecutivos no aportan nada y se rechazan (`[A, +, +, D]`); usá uno solo.
- `_` y `+` son **reservados**: si una zona del JSON se llama `_` o `+`, no se puede usar como literal.
- Los literales distinguen mayúsculas y se comparan exactos: `A` no matchea la zona `a`.
- Un patrón de un solo literal es válido: `[A]` matchea únicamente trayectorias de exactamente una zona, `A`.
- Un patrón sin literales se rechaza (`[_]`, `[_, _]`, `[+, +]`): matchearía casi cualquier trayectoria y volvería inútil al resto de la config.
- Repetir una etiqueta es válido y significativo: `[A, B, A]` matchea un vehículo que vuelve a `A`.

Para entender los casos límite: la trayectoria se serializa con comas (`A,B,D`), cada patrón se traduce a regex (`_` → `[^,]+`, `+` → `[^,]+(?:,[^,]+)*`, literales escapados) y se compara con `fullmatch`. Por eso los comodines nunca cruzan el separador de zonas.

**Match exacto**

| Patrón | Trayectoria | ¿Match? | Por qué |
|---|---|---|---|
| `[A, B, C]` | `[A, B, C]` | ✅ | cubre todo |
| `[A, B, C]` | `[A, B, C, D]` | ❌ | sobra `D` |
| `[A, B, C]` | `[A, B]` | ❌ | falta `C` |
| `[B, C]` | `[A, B, C]` | ❌ | sobra `A` al inicio |
| `[A, C]` | `[A, B, C]` | ❌ | falta `B` en el medio |
| `[A, B]` | `[B, A]` | ❌ | el orden importa |
| `[A, B, A]` | `[A, B, A]` | ✅ | reingreso a `A` |

**Comodín `_` (una zona)**

| Patrón | Trayectoria | ¿Match? | Por qué |
|---|---|---|---|
| `[A, _, C]` | `[A, B, C]` | ✅ | `_` = `B` |
| `[A, _, C]` | `[A, B, D, C]` | ❌ | `_` es una sola zona |
| `[A, _, C]` | `[A, C]` | ❌ | `_` no matchea cero |
| `[A, _, C]` | `[A, B, C, D]` | ❌ | sobra `D` |
| `[A, _, _]` | `[A, B, C]` | ✅ | dos zonas cualesquiera |
| `[A, _, _]` | `[A, B]` | ❌ | falta una |
| `[_, B]` | `[A, B]` | ✅ | `_` = `A` |
| `[_, B]` | `[A, X, B]` | ❌ | `_` es una sola |

**Comodín `+` (una o más)**

| Patrón | Trayectoria | ¿Match? | Por qué |
|---|---|---|---|
| `[A, +]` | `[A, B]` | ✅ | `+` = `B` |
| `[A, +]` | `[A, B, C, D]` | ✅ | `+` = `B, C, D` |
| `[A, +]` | `[A]` | ❌ | `+` no matchea cero |
| `[A, +]` | `[B, A]` | ❌ | no arranca en `A` |
| `[+, D]` | `[A, D]` | ✅ | `+` = `A` |
| `[+, D]` | `[A, B, C, D]` | ✅ | `+` = `A, B, C` |
| `[+, D]` | `[D]` | ❌ | `+` no matchea cero |
| `[+, D]` | `[D, A]` | ❌ | no termina en `D` |
| `[L, +, D]` | `[L, C, D]` | ✅ | `+` = `C` |
| `[L, +, D]` | `[L, M, N, D]` | ✅ | `+` = `M, N` |
| `[L, +, D]` | `[L, D]` | ❌ | `+` no matchea cero |
| `[+, B, A]` | `[X, Y, B, A]` | ✅ | `+` = `X, Y` |
| `[+, B, A]` | `[B, A]` | ❌ | falta al menos una antes de `B` |
| `[A, +, D]` | `[A, B, C, D]` | ✅ | `+` = `B, C` |
| `[A, +, D]` | `[A, D]` | ❌ | `+` no matchea cero |

**Combinados**

| Patrón | Trayectoria | ¿Match? | Por qué |
|---|---|---|---|
| `[A, _, +, D]` | `[A, X, Y, D]` | ✅ | `_` = `X`, `+` = `Y` |
| `[A, _, +, D]` | `[A, X, Y, Z, D]` | ✅ | `_` = `X`, `+` = `Y, Z` |
| `[A, _, +, D]` | `[A, X, D]` | ❌ | `+` no matchea cero |

##### Precedencia: quién gana cuando matchean varios

Todos los patrones de todos los grupos se compilan juntos y se ordenan una sola vez. Para cada trayectoria se recorre esa lista y gana el **primer patrón que matchea y cuyo grupo acepta el tipo de vehículo**. Ese orden no es el del YAML: el YAML solo desempata al final.

Criterios, en orden:

1. **Más zonas literales.** Un patrón 100 % definido le gana a uno con comodines.
2. **Menos `+`.** A igualdad de literales, gana el más acotado.
3. **Más `_`.** A igualdad de literales y de `+`, más `_` significa un patrón más largo y por lo tanto más restrictivo.
4. **Orden del YAML**: primero el grupo declarado antes; dentro de un grupo, el patrón declarado antes.

| Patrones que matchean | Trayectoria | Gana | Por qué |
|---|---|---|---|
| `[A, B, C, D]` vs `[+, D]` | `[A, B, C, D]` | `[A, B, C, D]` | 4 literales vs 1 |
| `[A, B, +]` vs `[A, _, _]` | `[A, B, C]` | `[A, B, +]` | 2 literales vs 1 |
| `[A, B, +]` vs `[A, B, _, _]` | `[A, B, C, D]` | `[A, B, _, _]` | 2 literales, 1 `+` vs 0 |
| `[A, _, +]` vs `[A, +]` | `[A, B, C]` | `[A, _, +]` | 1 literal, 1 `+`, más `_` |
| `[_, _]` vs `[A, B]` | `[A, B]` | `[A, B]` | 2 literales vs 0 |
| `[A, +]` vs `[+, D]` | `[A, B, D]` | el primero del YAML | empate total |
| `[A, B, +]` vs `[A, +, D]` | `[A, B, C, D]` | el primero del YAML | empate total |

Detalles que importan:

- La comparación es **por patrón**, no por grupo. Los patrones de un grupo compiten contra los de todos los demás.
- El filtro de tipo se evalúa antes: si el patrón más específico pertenece a un grupo que excluye ese tipo de vehículo, se saltea y gana el siguiente patrón que sí matchea.
- Un empate real es una config ambigua: el resultado depende de dónde esté declarado cada grupo. Si dos flujos pueden matchear la misma trayectoria, hacelos distinguibles o aceptá explícitamente el orden.

##### Sin match: `UNKNOWN`

Si ningún patrón matchea (o el que matchea pertenece a un grupo que excluye ese tipo), la trayectoria queda sin flujo:

- En `detections.csv` aparece con `path_asignado = UNKNOWN` y la columna `patron` vacía.
- No entra en `aggregated_counts.csv` ni en el gráfico: solo se cuentan las trayectorias matcheadas.
- Se agrega a `unmatched_paths.csv` para que puedas ajustar la config.

##### Validaciones al cargar el YAML

Todas se evalúan antes de procesar nada. Si falla una, el CLI imprime el error y sale con código `1`:

| Caso | Ejemplo | Resultado |
|---|---|---|
| Clave desconocida en `count:` | `count: {grups: []}` | Error con la lista de claves válidas |
| Clave desconocida en un grupo | `zones: ...` escrito `zone:` | Error indicando el grupo y las claves válidas |
| Grupo que no es un mapa | `groups: [flujo]` | Error de tipo |
| Grupo sin `zones` o con `zones: []` | `{name: "1", access: sur}` | Error |
| Patrón vacío | `- []` | Error |
| Zona vacía | `- [A, " ", B]` | Error |
| Patrón sin literales | `- [_]`, `- [_, _]`, `- [+, _]` | Error: matchearía cualquier trayectoria |
| Dos `+` consecutivos | `- [A, +, +, B]` | Error |
| Zona con coma | `- [A,B]` | Error (la coma es el separador interno) |
| Patrón repetido | `[A, B]` en dos grupos (o dos veces en el mismo) | Error: cada trayectoria debe matchear con un solo flujo |
| Comodín `*` o `%` | `- [*, B]` | Error que sugiere `_` y `+` |

##### Trampas de YAML

- `_` y `+` se escriben sin comillas: YAML los trata como texto. `*` y `%` **no** sirven como comodines porque sin comillas `*` es un alias y `%` no puede empezar un token.
- Las etiquetas que YAML interpreta como otro tipo hay que citarlas. `on`, `off`, `yes`, `no`, `true`, `false` se vuelven booleanos; `123` se vuelve número; `~` se vuelve `null`. El código stringifica lo que llegue, así que `[ON, OFF]` termina siendo `["True", "False"]` y no matchea nunca. Escribí `["ON", "OFF"]`.
- La coma separa elementos de la lista YAML, por eso no puede formar parte de una etiqueta de zona. Si tu zona se llama `A,1`, renombrala en el JSON.
- Los espacios después de las comas son opcionales (`[A,F]` y `[A, F]` son equivalentes).

##### Archivos que genera

**`detections.csv`** — una fila por trayectoria, matcheada o no:

| Columna | Contenido |
|---|---|
| `vehicle_id` | ID compuesto `video_id_tipo` |
| `vehicle_type` | Tipo de vehículo |
| `zones` | Trayectoria, con formato `[A, F]` |
| `hora_inicial` / `hora_final` | Primera y última entrada a una zona (`HH:MM:SS`) |
| `path_asignado` | Nombre del flujo o `UNKNOWN` |
| `patron` | Patrón que matcheó, con formato `[A,F]`; vacío si es `UNKNOWN` |

```csv
vehicle_id,vehicle_type,zones,hora_inicial,hora_final,path_asignado,patron
26_12-11-2026_17.45.00_10001_car,car,"[A, F]",17:45:00,17:45:02,1,"[A,F]"
26_12-11-2026_17.45.00_10002_car,car,"[B]",17:45:03,17:45:04,UNKNOWN,
```

**`unmatched_paths.csv`** — las trayectorias sin flujo, agrupadas por secuencia exacta de zonas y tipo, ordenadas de mayor a menor frecuencia:

| Columna | Contenido |
|---|---|
| `zones` | Secuencia de zonas |
| `vehicle_type` | Tipo de vehículo |
| `count` | Cantidad de trayectorias con esa secuencia y tipo |
| `hora_min` / `hora_max` | Primer y último avistamiento del grupo |

```csv
zones,vehicle_type,count,hora_min,hora_max
[B],motorcycle,25001,05:59:54,19:59:58
[A],person,16221,06:00:00,19:58:36
```

Es el insumo para ajustar los patrones: mirá qué secuencias se repiten y agregá `_`/`+` donde corresponda. Si en una corrida todo matchea, el archivo se borra para no dejar datos viejos.

**`aggregated_counts.csv`** — matriz con **tres filas de encabezado**: acceso, flujo y tipo de vehículo. La primera columna es el intervalo; cada celda es el conteo. El conteo de una trayectoria se asigna al intervalo de su **primera** entrada a una zona.

```csv
intervalo,norte,,,sur,,
,Flujo Norte,,,Flujo Sur,,
,car,truck,bicycle,car,bus,truck
06:00-06:15,12,3,0,8,1,2
06:15-06:30,9,1,1,6,0,1
```

El archivo se escribe con BOM UTF-8 (lo abre bien Excel) e incluye todos los intervalos del día, incluso los que quedan en cero. Los accesos salen en el orden `norte → sur → oeste → este`; dentro de cada acceso, los flujos en el orden del YAML y los tipos según `include` (o en el orden del modelo cuando es `all`).

**`aggregated_chart.html` / `.png`** — un gráfico por grupo, en el orden del YAML, con barras apiladas por tipo de vehículo. `web` genera HTML con Chart.js desde el CDN de jsDelivr (necesita internet para verse); `image` genera PNG con Plotly + Kaleido; `none` no genera gráfico.

##### Diagnóstico de patrones muertos

Al terminar, `afora group` imprime los patrones que nunca ganaron una trayectoria:

```
Patrones sin coincidencias:
  92 [E,B,D]
  31 [F,A,E,B]
```

Significa que ese patrón no aportó ningún conteo. Las causas habituales:

1. La secuencia no aparece en los datos (la más común).
2. El `include`/`exclude` del grupo deja afuera a todos los vehículos que recorren esa secuencia.
3. Otro patrón más específico absorbe todos sus casos.

No es un error de sintaxis: el patrón es válido, simplemente no está aportando. Si esperabas conteos de ese flujo, es la primera pista para revisar la config.

##### Migración desde el match por subsecuencia

Si venías de una config vieja, el matcheo cambió: antes alcanzaba con que el patrón apareciera **en algún lugar** de la trayectoria; ahora tiene que cubrirla entera. Estas son las diferencias que más se notan:

| Antes (subsecuencia) | Ahora (match completo) |
|---|---|
| `[A, B]` matcheaba cualquier trayectoria que pasara por `A` y después por `B`, con zonas libres antes, en el medio o después | solo matchea exactamente `[A, B]` |
| `[A]` matcheaba cualquier trayectoria que tocara `A` | solo matchea trayectorias de una sola zona, `A` |
| Ganaba el `zone_path` más largo | Gana el más específico: más literales → menos `+` → más `_` → orden del YAML |
| Casi toda trayectoria entraba en el flujo con el patrón más largo que apareciera como subsecuencia | Solo entra si un patrón cubre la trayectoria completa; si no, queda `UNKNOWN` y se reporta en `unmatched_paths.csv` |

Receta para migrar:

1. Corré `afora group` con la config actual y abrí `unmatched_paths.csv`.
2. Para las secuencias cortas que sí querés contar (por ejemplo `[B]`), agregá el patrón explícito `[B]`.
3. Para variantes con prefijo o sufijo variable, usá `+`: `[+, B, A]`, `[B, A, +]`, `[A, +, D]`.
4. Para una zona intermedia cualquiera, usá `_`: `[A, _, D]`.
5. Volvé a correr y revisá la lista de patrones muertos y el top de `unmatched_paths.csv`.


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

**Exit codes:** `0` si todos los videos se procesaron correctamente (o no habia pendientes); `1` si algun video fallo. El mensaje final incluye el recuento: `Completados N videos (X con error), ... detecciones totales`. Los errores se imprimen a **stdout** (no a stderr) en entornos sin TTY como Colab, para que siempre queden visibles.

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

**Qué archivos lee.** Solo los `*_detections.csv` que estén **directamente** en `--folder` (no baja a subcarpetas) y cuyo nombre tenga el patrón `..._YYYY-MM-DD_HH.MM.SS_detections.csv` o `..._DD-MM-YYYY_HH.MM.SS_detections.csv`. Los archivos que no cumplen el formato se saltean en silencio, así que si el conteo da cero revisá primero los nombres. Para procesar varias fechas, corré el comando una vez por carpeta o usá `afora run`, que lo hace automáticamente por subdirectorio.

**Qué genera.** `detections.csv` (todas las trayectorias con flujo y patrón), `unmatched_paths.csv` (las que no matchearon, agrupadas por secuencia y tipo con conteo y rango horario), `aggregated_counts.csv` y el gráfico según `graph:`. Al final imprime los patrones que no matchearon nada y los archivos generados. Si el YAML no tiene sección `count:`, sale con código `1` sin procesar.

Ver [Conteo y gráficos](#conteo-y-gráficos) para la sintaxis completa de `count:`, patrones, precedencia y formato de cada archivo.

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
   - Genera `detections.csv` con intervenciones (paths asignados + patrón usado)
   - Genera `unmatched_paths.csv` con las trayectorias que no matchearon ningún flujo
   - Genera `aggregated_counts.csv` con conteo por acceso → flujo → tipo → intervalo
   - Genera `aggregated_chart.html` con gráficos de barras apiladas por flujo (Chart.js)

### 4. Resultados

```
C:\output\cam3\
├── 2026-06-01\
│   ├── cam3_20260601143000_01.mp4        # video procesado
│   ├── cam3_20260601143000_01_detections.csv  # detecciones crudas
│   ├── detections.csv                    # intervenciones (paths asignados + patrón)
│   ├── unmatched_paths.csv               # trayectorias sin flujo (si hay)
│   ├── aggregated_counts.csv             # conteo agregado
│   └── aggregated_chart.html             # gráfico interactivo
├── 2026-06-02\
│   ├── cam3_20260602120000_03.mp4
│   ├── cam3_20260602120000_03_detections.csv
│   ├── detections.csv
│   ├── unmatched_paths.csv
│   ├── aggregated_counts.csv
│   └── aggregated_chart.html
└── progress.csv                          # control de reanudación
```

### 5. Ver resultados

Abrir `aggregated_chart.html` en el navegador para ver los gráficos interactivos por flujo, o leer el CSV. `aggregated_counts.csv` tiene tres filas de encabezado (acceso, flujo, tipo de vehículo) y una fila por intervalo:

```csv
intervalo,norte,,,sur,,
,Flujo Norte,,,Flujo Sur,,
,car,truck,bicycle,car,bus,truck
06:00-06:15,12,3,0,8,1,2
06:15-06:30,9,1,1,6,0,1
```

Para auditar por qué cada vehículo cayó en cada flujo, `detections.csv` trae la columna `patron`. Y `unmatched_paths.csv` muestra las secuencias que no matchearon, ordenadas por frecuencia: es el punto de partida para agregar `_`/`+` a la config.

### Resumen del pipeline de conteo

```
detections.csv (crudas)
  → AbsoluteTimeConverter (tiempo absoluto + composite vehicle_id)
  → VehiclePathBuilder (trayectorias por vehículo)
  → FlowMatcher (match completo con _ / + y precedencia por especificidad)
  → InterventionFileExporter (detections.csv con patrón + unmatched_paths.csv)
  → IntervalGrouper (agregación por intervalos)
  → aggregated_counts.csv + aggregated_chart.{png|html}
```
