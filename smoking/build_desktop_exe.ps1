param(
    [string]$EntryPoint = "desktop_app.py",
    [string]$ExeName = "StealthLensDesktop"
)

$ErrorActionPreference = "Stop"

Write-Host "Atualizando pip..."
python -m pip install --upgrade pip

Write-Host "Instalando dependências do projeto..."
python -m pip install -r requirements.txt

Write-Host "Instalando PyInstaller..."
python -m pip install pyinstaller

Write-Host "Limpando artefatos anteriores de build..."
$workspace = (Get-Location).Path
$buildTarget = Join-Path $workspace "build\$ExeName"
$distTarget = Join-Path $workspace "dist\$ExeName"
foreach ($target in @($buildTarget, $distTarget)) {
    if (-not (Test-Path $target)) {
        continue
    }
    try {
        Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction Stop
    }
    catch {
        Write-Warning "Não foi possível limpar '$target'. O build continuará usando artefatos existentes."
    }
}

Write-Host "Gerando build .exe (modo onedir para maior compatibilidade com OpenCV/Torch)..."
$pyinstallerArgs = @(
    "--noconfirm",
    "--clean",
    "--windowed",
    "--onedir",
    "--name", $ExeName,
    "--collect-data", "ultralytics",
    "--collect-submodules", "ultralytics"
)

if (Test-Path "config.example.json") {
    $pyinstallerArgs += @("--add-data", "config.example.json;.")
}

$modelFiles = @(
    "runs/detect/train/weights/best.pt",
    "yolov8m.pt",
    "yolo26n.pt"
)

foreach ($modelFile in $modelFiles) {
    if (-not (Test-Path $modelFile)) {
        continue
    }

    $destination = Split-Path $modelFile -Parent
    if ([string]::IsNullOrWhiteSpace($destination)) {
        $destination = "."
    }
    $destination = $destination.Replace("\", "/")
    $pyinstallerArgs += @("--add-data", "$modelFile;$destination")
}

$pyinstallerArgs += $EntryPoint
pyinstaller @pyinstallerArgs
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller falhou com código $LASTEXITCODE."
}

Write-Host "Build concluído."
Write-Host "Executável: .\\dist\\$ExeName\\$ExeName.exe"
