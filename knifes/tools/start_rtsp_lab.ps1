param(
    [int]$Port = 8554,
    [string]$LabDir = "artifacts\rtsp-lab"
)

$ErrorActionPreference = "Stop"

function Assert-Command($Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Comando '$Name' nao encontrado no PATH. Instale-o antes de iniciar o laboratorio RTSP."
    }
}

Assert-Command "mediamtx"
Assert-Command "ffmpeg"

$labPath = Join-Path (Get-Location) $LabDir
New-Item -ItemType Directory -Force -Path $labPath | Out-Null
Remove-Item -Path (Join-Path $labPath "*.log") -Force -ErrorAction SilentlyContinue

Get-Process ffmpeg,mediamtx -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

$configPath = Join-Path $labPath "mediamtx.yml"
@"
logLevel: info
rtspAddress: :$Port
rtspTransports: [tcp]

paths:
  cam1:
    source: publisher
  cam2:
    source: publisher
  cam3:
    source: publisher
  cam4:
    source: publisher
"@ | Set-Content -Encoding UTF8 $configPath

$server = Start-Process mediamtx `
    -ArgumentList $configPath `
    -WorkingDirectory $labPath `
    -RedirectStandardOutput (Join-Path $labPath "mediamtx.log") `
    -RedirectStandardError (Join-Path $labPath "mediamtx.err.log") `
    -PassThru `
    -WindowStyle Hidden

Start-Sleep -Seconds 2

$filters = @(
    "testsrc2=size=640x360:rate=10",
    "smptebars=size=640x360:rate=10",
    "testsrc=size=640x360:rate=10",
    "color=c=steelblue:size=640x360:rate=10"
)

$publishers = @()
for ($i = 1; $i -le 4; $i++) {
    $args = @(
        "-hide_banner",
        "-loglevel", "warning",
        "-re",
        "-f", "lavfi",
        "-i", $filters[$i - 1],
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-g", "10",
        "-keyint_min", "10",
        "-sc_threshold", "0",
        "-bf", "0",
        "-x264-params", "repeat-headers=1",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        "rtsp://127.0.0.1:$Port/cam$i"
    )

    $publishers += Start-Process ffmpeg `
        -ArgumentList $args `
        -WorkingDirectory $labPath `
        -RedirectStandardOutput (Join-Path $labPath "ffmpeg-cam$i.out.log") `
        -RedirectStandardError (Join-Path $labPath "ffmpeg-cam$i.err.log") `
        -PassThru `
        -WindowStyle Hidden
}

Start-Sleep -Seconds 3

Write-Host "Laboratorio RTSP iniciado."
Write-Host "MediaMTX PID: $($server.Id)"
Write-Host "FFmpeg PIDs: $($publishers.Id -join ', ')"
Write-Host "Endpoints:"
1..4 | ForEach-Object { Write-Host "  rtsp://127.0.0.1:$Port/cam$_" }
Write-Host "Logs: $labPath"
