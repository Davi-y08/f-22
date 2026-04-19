# Stealth Lens Agent

Agente local profissional para monitoramento de múltiplas câmeras RTSP com visão computacional usando Ultralytics YOLO.

## O que o projeto faz

- Conecta em múltiplas câmeras RTSP simultaneamente
- Usa um worker por câmera
- Faz sampling de frames para controlar custo computacional
- Detecta eventos com múltiplos modelos YOLO por câmera
- Salva eventos em JSONL e snapshots opcionais
- Mantém status operacional por câmera em arquivo JSON
- Reconecta automaticamente quando um stream cai
- Mostra a câmera ao vivo com overlays de detecção e métricas
- Confirma tabagismo por associação temporal entre `person`, `cigarette` e `smoke`
- Descobre automaticamente câmeras na rede por ONVIF/RTSP e permite escolher no terminal

## Estrutura

```text
main.py
desktop_app.py
build_desktop_exe.ps1
agent/
  manager.py
  worker.py
discovery/
  service.py
events/
  emitter.py
models/
  loader.py
streams/
  rtsp_client.py
utils/
  config.py
  logger.py
legacy/
  smoking_demo.py
```

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Se você pretende usar GPU NVIDIA, instale antes a versão correta do PyTorch/CUDA compatível com seu ambiente.

## Configuração

O `main.py` carrega `config.json` por padrão. Neste momento ele vem pronto para teste local com:

- `source: 0` para abrir a câmera local
- `display.enabled: true` para mostrar a janela ao vivo
- `runs/detect/train/weights/best.pt` como modelo de `person`, `cigarette` e `smoke`
- `smoking_behavior.enabled: true` para confirmar evento de fumo só após persistência temporal por pessoa rastreada

Depois ajuste conforme necessário:

- `model_catalog`: define caminhos, classes e thresholds por modelo
- `cameras`: define RTSP, FPS de análise, zonas e modelos ativos por câmera
- `smoking_behavior`: define a heurística profissional de tabagismo por tracking e associação espacial
- `storage`: define onde eventos, snapshots e status serão gravados

Para usar uma câmera RTSP/IP, troque `source` para a URL RTSP e use o [config.example.json](</C:/Users/oisyz/OneDrive/Desktop/projects/f-22/smoking/config.example.json>) como base.

## Execução

```bash
python main.py
```

Para usar a interface desktop (GUI):

```bash
python desktop_app.py
```

Para descobrir câmeras IP/local automaticamente e escolher uma no terminal:

```bash
python main.py --discover
```

Ou usando outro arquivo:

```bash
python main.py --config config.example.json
```

Pressione `q` ou `Esc` na janela para encerrar o monitoramento visual.

## Executável .exe (Windows)

Para gerar o executável da interface desktop:

```powershell
.\build_desktop_exe.ps1
```

Saída esperada:

- `dist\StealthLensDesktop\StealthLensDesktop.exe`

Observações:

- O build usa `--onedir` por estabilidade com bibliotecas de visão computacional.
- O `.exe` abre a GUI para descobrir câmeras, salvar no `config.json` e iniciar/parar monitoramento.

## Saídas

- Eventos: `artifacts/events/events-YYYY-MM-DD.jsonl`
- Snapshots: `artifacts/snapshots/<camera_id>/`
- Status: `artifacts/status/agent-status.json`

## Observações de produto

- O backend tenta usar `PyAV` primeiro quando disponível para RTSP por ser mais robusto, com fallback para OpenCV/FFmpeg
- Cada worker controla cooldown de eventos para evitar tempestade de alertas
- Zonas podem ser definidas em coordenadas normalizadas de `0.0` a `1.0`
- O sistema está estruturado para crescer para APIs, filas externas, painel web e multi-tenant no futuro
