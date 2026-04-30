param(
    [string]$EntryPoint = "desktop_app.py",
    [string]$ExeName = "StealthLensKnifeDesktopLite",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $venvPython = Join-Path (Get-Location).Path ".venv-lite\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $PythonExe = $venvPython
    } else {
        $PythonExe = "python"
    }
}

Write-Host "Python em uso: $PythonExe"

Write-Host "Atualizando pip..."
& $PythonExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Falha ao atualizar pip."
}

$pythonVersion = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) {
    throw "Falha ao detectar versão do Python."
}
Write-Host "Python detectado: $pythonVersion"

Write-Host "Instalando dependências Lite..."
& $PythonExe -m pip install -r requirements-lite.txt
if ($LASTEXITCODE -ne 0) {
    throw "Falha ao instalar dependências Lite. Em geral, use Python 3.10, 3.11 ou 3.12 para maior compatibilidade."
}

Write-Host "Instalando PyInstaller..."
& $PythonExe -m pip install pyinstaller
if ($LASTEXITCODE -ne 0) {
    throw "Falha ao instalar PyInstaller."
}

$workspace = (Get-Location).Path
$buildStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$tempRoot = Join-Path $env:TEMP "stealth-lens-build-$buildStamp"
$stageBuildDir = Join-Path $tempRoot "build"
$stageDistRoot = Join-Path $tempRoot "dist"
Write-Host "Build isolado em: $tempRoot"

Write-Host "Gerando build Lite .exe (CPU-first, sem runtime full GPU embutido)..."
if (-not (Test-Path "config.lite.example.json")) {
    throw "Arquivo obrigatório não encontrado: config.lite.example.json"
}
if (-not (Test-Path "models/knife_monitor.onnx")) {
    throw "Modelo Lite ausente em models/knife_monitor.onnx. Rode: python export_lite_model.py"
}

$pyinstallerArgs = @(
    "--noconfirm",
    "--clean",
    "--windowed",
    "--onedir",
    "--workpath", $stageBuildDir,
    "--distpath", $stageDistRoot,
    "--name", $ExeName
)

$pyinstallerArgs += @("--add-data", "config.lite.example.json;.")
$pyinstallerArgs += @("--add-data", "models/knife_monitor.onnx;models")
$pyinstallerArgs += @("--hidden-import", "onnxruntime")
$pyinstallerArgs += @("--collect-binaries", "onnxruntime")
$pyinstallerArgs += @("--collect-data", "onnxruntime")
$pyinstallerArgs += @("--exclude-module", "torch")
$pyinstallerArgs += @("--exclude-module", "torchvision")
$pyinstallerArgs += @("--exclude-module", "ultralytics")
$pyinstallerArgs += @("--exclude-module", "matplotlib")
$pyinstallerArgs += @("--exclude-module", "scipy")

$pyinstallerArgs += $EntryPoint
& $PythonExe -m PyInstaller @pyinstallerArgs
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller falhou com código $LASTEXITCODE."
}

$workspaceStageDist = Join-Path $workspace "dist\_stage_$ExeName`_$buildStamp"
if (Test-Path $workspaceStageDist) {
    Remove-Item -LiteralPath $workspaceStageDist -Recurse -Force
}
New-Item -ItemType Directory -Path $workspaceStageDist | Out-Null
Copy-Item -Path (Join-Path (Join-Path $stageDistRoot $ExeName) "*") -Destination $workspaceStageDist -Recurse -Force

$distAppDir = $workspaceStageDist

$distModelsDir = Join-Path $distAppDir "models"
if (-not (Test-Path $distModelsDir)) {
    New-Item -ItemType Directory -Path $distModelsDir | Out-Null
}

$distModelPath = Join-Path $distModelsDir "knife_monitor.onnx"
$internalModelPath = Join-Path $distAppDir "_internal\models\knife_monitor.onnx"
if (Test-Path $internalModelPath) {
    Copy-Item -LiteralPath $internalModelPath -Destination $distModelPath -Force
} elseif (-not (Test-Path $distModelPath)) {
    Copy-Item -LiteralPath "models/knife_monitor.onnx" -Destination $distModelPath -Force
}

$liteConfig = Join-Path $distAppDir "config.lite.example.json"
$internalLiteConfig = Join-Path $distAppDir "_internal\config.lite.example.json"
if (-not (Test-Path $liteConfig) -and (Test-Path $internalLiteConfig)) {
    Copy-Item -LiteralPath $internalLiteConfig -Destination $liteConfig -Force
}
$targetConfig = Join-Path $distAppDir "config.json"
if (Test-Path $liteConfig) {
    Copy-Item -LiteralPath $liteConfig -Destination $targetConfig -Force
}

$portableZip = ".\dist\$ExeName-portable-$buildStamp.zip"
if (Test-Path $portableZip) {
    Remove-Item -LiteralPath $portableZip -Force
}
Compress-Archive -Path "$distAppDir\*" -DestinationPath $portableZip

Write-Host "Build Lite concluído."
Write-Host "Executável: $distAppDir\$ExeName.exe"
Write-Host "Pacote para compartilhamento: $portableZip"

