# Stealth Lens Knife Agent

Agente local profissional para monitoramento de cameras e deteccao de faca com visao computacional.

## Recursos

- Interface desktop para descobrir webcams locais, RTSP, HTTP/IP Webcam e cameras ONVIF/RTSP.
- Suporte a multiplas cameras simultaneas.
- Worker separado por camera com reconexao, fila de frames e limite de FPS de analise.
- Modelo `knife_monitor` configurado para detectar a classe `knife`.
- Eventos em JSON e snapshots quando uma faca for detectada.
- Build Lite distribuivel com ONNX Runtime, sem empacotar Torch/Ultralytics.

## Arquivos importantes

- `legacy/main_knife_legacy.py`: versao antiga preservada.
- `runs/detect/train/weights/best.pt`: modelo treinado original em PyTorch/Ultralytics.
- `models/knife_monitor.onnx`: modelo Lite exportado para distribuicao.
- `desktop_app.py`: interface desktop.
- `config.lite.example.json`: configuracao usada no executavel Lite.

## Rodar no modo desenvolvimento

Use um Python que tenha as dependencias full instaladas. Nesta maquina, o Python 3.13 ja possui Ultralytics/Torch.

```powershell
py -3.13 -m pip install -r requirements.txt
py -3.13 desktop_app.py
```

Para rodar o agente direto pelo terminal:

```powershell
py -3.13 main.py --config config.json
```

## Gerar modelo Lite ONNX

```powershell
py -3.13 export_lite_model.py
```

O arquivo gerado fica em `models/knife_monitor.onnx`.

## Gerar pacote distribuivel

Crie/ative um ambiente Lite com Python 3.11 ou 3.12:

```powershell
py -3.11 -m venv .venv-lite
.\.venv-lite\Scripts\python.exe -m pip install --upgrade pip
.\.venv-lite\Scripts\python.exe -m pip install -r requirements-lite.txt
.\build_desktop_exe_lite.ps1
```

O pacote final fica em `dist\StealthLensKnifeDesktopLite-portable-*.zip`.

## Laboratorio RTSP com 4 cameras falsas

```powershell
.\tools\start_rtsp_lab.ps1
```

Endpoints criados:

```text
rtsp://127.0.0.1:8554/cam1
rtsp://127.0.0.1:8554/cam2
rtsp://127.0.0.1:8554/cam3
rtsp://127.0.0.1:8554/cam4
```

Para parar:

```powershell
.\tools\stop_rtsp_lab.ps1
```

## Observacao importante

Deteccao de faca e um caso sensivel. Use com validacao humana, thresholds conservadores e testes reais no ambiente do cliente antes de qualquer uso operacional.
