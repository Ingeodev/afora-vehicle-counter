<#
.SYNOPSIS
    Instala afora-vehicle-counter en un entorno virtual dedicado con deteccion de CUDA.
.DESCRIPTION
    - Detecta la version de CUDA instalada y selecciona la variante de PyTorch adecuada.
    - Crea un .venv en %PROGRAMDATA%\afora-vehicle-counter (evita redireccion de Windows Store Python).
    - Instala el wheel con las dependencias opcionales correspondientes.
    - Agrega el directorio Scripts del venv al PATH del usuario (permanente).
    - Establece la variable de entorno AFORA_VC_HOME.
#>

$ErrorActionPreference = "Stop"

# Usar PROGRAMDATA en vez de LOCALAPPDATA para evitar la redireccion
# de Windows Store Python sobre %APPDATA% y %LOCALAPPDATA%
$REPO_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV_DIR = "$env:PROGRAMDATA\afora-vehicle-counter"
$WHEEL_DIR = "$REPO_DIR\dist"
$WHEEL_FILE = Get-ChildItem -Path "$WHEEL_DIR\*.whl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (-not $WHEEL_FILE) {
    Write-Error "No se encontro ningun .whl en $WHEEL_DIR. Ejecuta 'python -m build --wheel' primero."
    exit 1
}

Write-Host "=== Instalacion de afora-vehicle-counter ===" -ForegroundColor Cyan
Write-Host "Wheel:    $($WHEEL_FILE.Name)" -ForegroundColor White
Write-Host "Destino:  $VENV_DIR" -ForegroundColor White

# ── Detectar CUDA ──────────────────────────────────────────
$CUDA_VERSION = $null
$TORCH_EXTRA = "torch-auto"

try {
    $nvccOut = & "nvcc" "--version" 2>&1
    # nvcc output is multi-line (array); convertir a string unico para que -match poblaria $Matches
    $nvccText = "$nvccOut"
    if ($nvccText -match "release (\d+\.\d+)") {
        $CUDA_VERSION = $Matches[1]
        Write-Host "CUDA detectado: $CUDA_VERSION" -ForegroundColor Green
    }
} catch {
    Write-Host "CUDA no detectado, usando torch-auto" -ForegroundColor Yellow
}

if ($CUDA_VERSION) {
    $TORCH_EXTRA = switch -Wildcard ($CUDA_VERSION) {
        "12.6*"  { "torch-cu126" }
        "12.4*"  { "torch-cu124" }
        "12.1*"  { "torch-cu121" }
        "11.8*"  { "torch-cu118" }
        "13*"    { "torch-cu126" }
        default  { "torch-auto" }
    }
}
Write-Host "Extra PyTorch: $TORCH_EXTRA" -ForegroundColor Green

# ── Crear .venv ────────────────────────────────────────────
if (Test-Path "$VENV_DIR\Scripts\python.exe") {
    Write-Host "El .venv ya existe en $VENV_DIR" -ForegroundColor Yellow
} else {
    Write-Host "Creando entorno virtual en $VENV_DIR ..." -ForegroundColor Cyan
    python -m venv "$VENV_DIR"
}

$PYTHON = "$VENV_DIR\Scripts\python.exe"
$PIP = "$VENV_DIR\Scripts\pip.exe"

# ── Instalar wheel ─────────────────────────────────────────
Write-Host "Instalando wheel y dependencias ..." -ForegroundColor Cyan
$EXTRAS = "rtdetr,tracking,$TORCH_EXTRA"
& $PIP install --upgrade pip
& $PIP install "$($WHEEL_FILE.FullName)[$EXTRAS]"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Fallo la instalacion del wheel"
    exit 1
}

# ── Verificar ──────────────────────────────────────────────
Write-Host "Verificando instalacion ..." -ForegroundColor Cyan
& $PYTHON -c "import afora_vc; print('afora_vc OK')"
& $PYTHON -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

# ── Variables de entorno ───────────────────────────────────
Write-Host "Configurando variables de entorno ..." -ForegroundColor Cyan

$SCRIPTS_DIR = "$VENV_DIR\Scripts"
$AFORA_HOME = $VENV_DIR

$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*$SCRIPTS_DIR*") {
    $newPath = "$SCRIPTS_DIR;$currentPath"
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "  PATH del usuario actualizado: $SCRIPTS_DIR" -ForegroundColor Green
} else {
    Write-Host "  PATH ya contiene $SCRIPTS_DIR" -ForegroundColor Yellow
}

[Environment]::SetEnvironmentVariable("AFORA_VC_HOME", $AFORA_HOME, "User")
Write-Host "  AFORA_VC_HOME = $AFORA_HOME" -ForegroundColor Green

Write-Host ""
Write-Host "=== Instalacion completada ===" -ForegroundColor Cyan
Write-Host "Usar: afora --help" -ForegroundColor White
Write-Host "O:   & `"$PYTHON`" -m afora_vc --help" -ForegroundColor White
Write-Host ""
Write-Host "NOTA: Reinicia la terminal para que los cambios de PATH surtan efecto." -ForegroundColor Yellow
