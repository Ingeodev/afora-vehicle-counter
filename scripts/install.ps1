# Instalador automatico de afora-vehicle-counter
# Detecta CUDA y elige la variante optima de PyTorch

param(
    [string[]]$Backend = @("yolo", "rtdetr"),
    [switch]$DryRun
)

Write-Host "=" * 60
Write-Host "  afora-vehicle-counter - Instalador automatico" -ForegroundColor Cyan
Write-Host "=" * 60

Write-Host "`n  Backends solicitados: $($Backend -join ', ')"

# Detectar CUDA via nvidia-smi
$cudaVersion = $null
$cudaVariant = "torch-auto"
$indexUrl = $null

try {
    $nvidiaSmi = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $nvidiaSmi) {
        Write-Host "  GPU detectada: $($nvidiaSmi.Trim())"

        $cudaOut = & nvidia-smi 2>$null
        foreach ($line in $cudaOut) {
            if ($line -match "CUDA Version:\s*([0-9]+\.[0-9]+)") {
                $cudaVersion = $Matches[1]
                break
            }
        }
    }
}
catch {
    Write-Host "  nvidia-smi no disponible" -ForegroundColor Yellow
}

if ($cudaVersion) {
    Write-Host "  CUDA Version: $cudaVersion" -ForegroundColor Green

    $variantMap = @{
        "11.8" = @{ Variant = "torch-cu118"; Index = "https://download.pytorch.org/whl/cu118" }
        "12.1" = @{ Variant = "torch-cu121"; Index = "https://download.pytorch.org/whl/cu121" }
        "12.4" = @{ Variant = "torch-cu124"; Index = "https://download.pytorch.org/whl/cu124" }
        "12.6" = @{ Variant = "torch-cu126"; Index = "https://download.pytorch.org/whl/cu126" }
    }

    $majorMinor = $cudaVersion
    if ($variantMap.ContainsKey($majorMinor)) {
        $cudaVariant = $variantMap[$majorMinor].Variant
        $indexUrl = $variantMap[$majorMinor].Index
    }
    else {
        Write-Host "  CUDA $cudaVersion no tiene variante especifica -> usando torch generico de PyPI" -ForegroundColor Yellow
        Write-Host "  (CUDA $cudaVersion es compatible con CUDA 12.x via backward compatibility)" -ForegroundColor Yellow
    }
}
else {
        Write-Host "  CUDA no detectado -> torch-auto (PyPI con CUDA incluido si hay GPU)" -ForegroundColor Yellow
}

$extras = ($Backend + $cudaVariant) -join ','
$pipCmd = "pip install `"afora-vehicle-counter[$extras]`""
if ($indexUrl) {
    $pipCmd += " --extra-index-url $indexUrl"
}

Write-Host "`n  Comando:" -ForegroundColor Cyan
Write-Host "`n    $pipCmd`n"

if (-not $DryRun) {
    Write-Host "  Ejecutando..." -ForegroundColor Cyan
    Write-Host ""

    if ($indexUrl) {
        & pip install "afora-vehicle-counter[$extras]" --extra-index-url $indexUrl
    }
    else {
        & pip install "afora-vehicle-counter[$extras]"
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n  Instalacion completada" -ForegroundColor Green
    }
    else {
        Write-Host "`n  Error durante la instalacion" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}
else {
    Write-Host "  [dry-run] No se ejecuto el comando." -ForegroundColor Yellow
}
