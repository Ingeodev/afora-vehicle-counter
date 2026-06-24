<#
.SYNOPSIS
    Instala afora-vehicle-counter en un entorno virtual dedicado con deteccion de CUDA.
.DESCRIPTION
    - Detecta la version de CUDA instalada y selecciona la variante de PyTorch adecuada.
    - Crea un .venv en %%PROGRAMDATA%%\afora-vehicle-counter.
    - Instala desde el .whl (si existe en dist/) o desde source con pip install .
    - Agrega el directorio Scripts del venv al PATH del usuario (permanente).
    - Establece la variable de entorno AFORA_VC_HOME.
#>

$ErrorActionPreference = "Stop"

$REPO_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV_DIR = "$env:PROGRAMDATA\afora-vehicle-counter"
$PYTHON = "$VENV_DIR\Scripts\python.exe"
$PIP = "$VENV_DIR\Scripts\pip.exe"

Write-Host "=== Instalacion de afora-vehicle-counter ===" -ForegroundColor Cyan
Write-Host "Destino:  $VENV_DIR" -ForegroundColor White

# ── Detectar CUDA ──────────────────────────────────────────
$CUDA_VERSION = $null
$TORCH_EXTRA = "torch-auto"

try {
    $nvccOut = & "nvcc" "--version" 2>&1
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
if (Test-Path "$PYTHON") {
    Write-Host "El .venv ya existe en $VENV_DIR" -ForegroundColor Yellow
} else {
    Write-Host "Creando entorno virtual ..." -ForegroundColor Cyan
    python -m venv "$VENV_DIR"
}

# ── Instalar ───────────────────────────────────────────────
$WHEEL_FILE = Get-ChildItem -Path "$REPO_DIR\dist\*.whl" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1

Write-Host "Instalando dependencias ..." -ForegroundColor Cyan
& $PIP install --upgrade pip

$EXTRAS = "rtdetr,tracking,$TORCH_EXTRA"
if ($WHEEL_FILE) {
    Write-Host "Desde wheel: $($WHEEL_FILE.Name)" -ForegroundColor White
    & $PIP install "$($WHEEL_FILE.FullName)[$EXTRAS]"
} else {
    Write-Host "Desde source: pip install .[$EXTRAS]" -ForegroundColor White
    & $PIP install "$REPO_DIR[$EXTRAS]"
}

if ($LASTEXITCODE -ne 0) {
    Write-Error "Fallo la instalacion"
    exit 1
}

# ── Verificar ──────────────────────────────────────────────
Write-Host "Verificando instalacion ..." -ForegroundColor Cyan
& $PYTHON -c "import afora_vc; print('afora_vc OK')"
& $PYTHON -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

# ── Variables de entorno ───────────────────────────────────
Write-Host "Configurando variables de entorno ..." -ForegroundColor Cyan

$SCRIPTS_DIR = "$VENV_DIR\Scripts"

$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*$SCRIPTS_DIR*") {
    $newPath = "$SCRIPTS_DIR;$currentPath"
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "  PATH del usuario actualizado: $SCRIPTS_DIR" -ForegroundColor Green
} else {
    Write-Host "  PATH ya contiene $SCRIPTS_DIR" -ForegroundColor Yellow
}

[Environment]::SetEnvironmentVariable("AFORA_VC_HOME", $VENV_DIR, "User")
Write-Host "  AFORA_VC_HOME = $VENV_DIR" -ForegroundColor Green

Write-Host ""
Write-Host "=== Instalacion completada ===" -ForegroundColor Cyan
Write-Host "Usar: afora --help" -ForegroundColor White
Write-Host ""
Write-Host "NOTA: Reinicia la terminal para que los cambios de PATH surtan efecto." -ForegroundColor Yellow
