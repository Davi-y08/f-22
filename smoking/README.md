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
- Descobre automaticamente câmeras HTTP compatíveis com IP Webcam (ex.: `http://<ip>:8080/video`)

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
display/
  renderer.py
events/
  emitter.py
models/
  loader.py
runtime/
  tuning.py
streams/
  rtsp_client.py
utils/
  config.py
  logger.py
  redaction.py
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

### Perfil Lite (distribuição leve)

Para distribuição em PCs comuns (CPU-first), use:

```bash
pip install -r requirements-lite.txt
```

Esse perfil usa `ONNX Runtime` para reduzir bastante o tamanho do pacote final.

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

Para distribuição leve, use [config.lite.example.json](</C:/Users/oisyz/OneDrive/Desktop/projects/f-22/smoking/config.lite.example.json>) com modelo ONNX.

Se você já treinou `best.pt`, gere o modelo ONNX Lite com:

```bash
python export_lite_model.py
```

## Execução

```bash
python main.py
```

Para usar a interface desktop (GUI):

```bash
python desktop_app.py
```

Para rodar os testes rápidos de configuração e segurança:

```bash
python -m unittest discover -s tests
```

Para descobrir câmeras IP/local automaticamente e escolher uma no terminal:

```bash
python main.py --discover
```

Na GUI, em `URL Manual (RTSP/HTTP)`, você pode informar só a base do IP Webcam (ex.: `http://192.168.1.244:8080/`) que o agente tenta automaticamente rotas de stream comuns (`/video`, `/?action=stream`, etc.).

Ou usando outro arquivo:

```bash
python main.py --config config.example.json
```

Pressione `q` ou `Esc` na janela para encerrar o monitoramento visual.
Pressione `F` para alternar entre fullscreen forçado, janela normal e modo definido no config.
O padrão atual é janela livre (`display.fullscreen: false`).

## Executável .exe (Windows)

Para gerar o executável da interface desktop:

```powershell
.\build_desktop_exe.ps1
```

Saída esperada:

- `dist\StealthLensDesktop\StealthLensDesktop.exe`

### Build Lite para clientes (recomendado)

```powershell
.\build_desktop_exe_lite.ps1
```

Recomendação de ambiente para build Lite: Python `3.10`, `3.11` ou `3.12`.

Saídas esperadas:

- `dist\StealthLensDesktopLite\StealthLensDesktopLite.exe`
- `dist\StealthLensDesktopLite-portable.zip`

Observações:

- O build usa `--onedir` por estabilidade com bibliotecas de visão computacional.
- O `.exe` abre a GUI para descobrir câmeras, salvar no `config.json` e iniciar/parar monitoramento.
- Para compartilhar com clientes, envie o `.zip` gerado (não envie apenas o `.exe`).

## Saídas

- Eventos: `artifacts/events/events-YYYY-MM-DD.jsonl`
- Snapshots: `artifacts/snapshots/<camera_id>/`
- Status: `artifacts/status/agent-status.json`

## Observações de produto

- O backend tenta usar `PyAV` primeiro quando disponível para RTSP por ser mais robusto, com fallback para OpenCV/FFmpeg
- Cada worker controla cooldown de eventos para evitar tempestade de alertas
- Zonas podem ser definidas em coordenadas normalizadas de `0.0` a `1.0`
- O sistema está estruturado para crescer para APIs, filas externas, painel web e multi-tenant no futuro
- `display` agora suporta modo profissional por câmera: `fullscreen`, `fit_mode` (`contain`/`cover`/`stretch`), `interpolation` e `enhance`
- `display.target_fps` permite limitar FPS de renderização por câmera (recomendado `20-30` para estabilidade com 3+ câmeras)
- O runtime aplica tuning automático de threads (OpenCV/ONNX/Torch CPU) para reduzir travamentos em multi-câmera sem perda de qualidade de detecção
- O motor ONNX é compartilhado entre câmeras que usam o mesmo modelo, reduzindo memória e tempo de inicialização em instalações multi-câmera
- URLs RTSP com usuário/senha são mascaradas em logs e mensagens de erro
- Para múltiplas câmeras no mesmo host/NVR, o `id` de câmera é gerado a partir da `source` (stream) para evitar substituição indevida no `config.json`
