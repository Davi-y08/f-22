# Stealth Lens Knife Agent

Agente local profissional para monitoramento de cameras e deteccao de faca com visao computacional.

## Recursos

- Interface desktop para descobrir webcams locais, RTSP, HTTP/IP Webcam e cameras ONVIF/RTSP.
- Suporte a multiplas cameras simultaneas.
- Worker separado por camera com reconexao, fila de frames e limite de FPS de analise.
- Modelo `knife_monitor` configurado para detectar a classe `knife`.
- Eventos em JSON e snapshots quando uma faca for detectada.
- Sincronizacao com API/site usando chave do distribuido, incluindo imagem do alerta.
- Interface desktop com campos para URL da API, ID do agente e chave/codigo do site.
- Pipeline de camera com baixa latencia, descarte de frames antigos e FPS de analise adaptativo.
- Build Lite distribuivel com ONNX Runtime, sem empacotar Torch/Ultralytics.

## Arquivos importantes

- `legacy/main_knife_legacy.py`: versao antiga preservada.
- `runs/detect/train/weights/best.pt`: modelo treinado original em PyTorch/Ultralytics.
- `models/knife_monitor.onnx`: modelo Lite exportado para distribuicao.
- `desktop_app.py`: interface desktop.
- `config.lite.example.json`: configuracao usada no executavel Lite.

## Conexao com API/site

O bloco `cloud` controla o envio de cameras e alertas para o site:

```json
{
  "cloud": {
    "enabled": false,
    "api_base_url": "https://api-f22.onrender.com",
    "agent_access_key": "",
    "sync_discovered_cameras": true,
    "sync_events": true,
    "timeout_seconds": 8
  }
}
```

No desktop, cole a chave gerada no site em `Chave/codigo do site`. Com a chave preenchida, o app ativa o envio para a API e os eventos `knife_detected` passam a ser sincronizados com snapshot.

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

Defaults do perfil Lite:

- `queue_maxsize: 2` para priorizar frame recente.
- `display.target_fps: 24` para reduzir travamentos em PCs comuns.
- `input_size: 640` para manter o modelo previsivel entre desktop e Lite.

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

## Confiabilidade e diagnostico

- A exibicao usa os frames atuais enquanto uma unica analise por camera roda em paralelo. Nao ha fila crescente de inferencias.
- A fila de envio fica em `artifacts/events/cloud-outbox.sqlite3`. Falhas temporarias sao tentadas novamente com intervalo crescente, ate 60 segundos, inclusive apos reiniciar.
- As fotos permanecem na pasta de snapshots e sao carregadas somente na hora de enviar. Preserve essa pasta junto com a fila ao mover a instalacao.
- A fila separa os destinos por URL, agente e chave. Uma chave diferente nao envia os alertas da chave anterior.
- Respostas HTTP 400/413/415/422 e fotos ausentes ficam marcadas como erro para inspecao, sem bloquear os demais alertas. Nao ha reenvio automatico dessas rejeicoes.
- A interface mostra alertas pendentes, enviados nesta sessao e com erro; os mesmos dados ficam no campo `cloud` do arquivo de status.
- Fotos, configuracao e status sao gravados por substituicao atomica. O horario do alerta corresponde a captura do frame.
- O processamento ONNX preserva objetos de classes diferentes em sobreposicao e expira rastreamentos mesmo nos frames sem deteccoes.

O codigo da API no projeto `../api` tambem foi ajustado para reconhecer o mesmo evento reenviado e notificar uma unica vez. Publique essa atualizacao da API antes de utilizar o reenvio em producao; ela utiliza a chave primaria existente, sem nova tabela.

Validacao automatizada: `python -m unittest discover -s tests`. Os testes simulam desconexao, reinicio, erros de foto, falha de gravacao e inferencia lenta. A precisao final deve ser avaliada com imagens reais das cameras.

## Observacao importante

Deteccao de faca e um caso sensivel. Use com validacao humana, thresholds conservadores e testes reais no ambiente do cliente antes de qualquer uso operacional.
