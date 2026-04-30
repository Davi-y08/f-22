$ErrorActionPreference = "Stop"

$processes = Get-Process ffmpeg,mediamtx -ErrorAction SilentlyContinue
if (-not $processes) {
    Write-Host "Nenhum processo RTSP lab encontrado."
    exit 0
}

$processes | Stop-Process -Force
Write-Host "Laboratorio RTSP parado."
