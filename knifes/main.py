from ultralytics import YOLO
import cv2
import time
from collections import deque, defaultdict

# =========================
# CONFIGURAÇÕES
# =========================
MODEL_PATH = "./runs/detect/train3/weights/best.pt"
CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Classe alvo
KNIFE_CLASS_ID = 1

# Inferência
CONF_THRESHOLD = 0.60
IOU_THRESHOLD = 0.45
IMG_SIZE = 640             # menor = mais rápido / maior = mais preciso
USE_HALF = True            # melhor em GPU compatível
DEVICE = 0                 # 0 = primeira GPU, "cpu" = CPU

# Rastreamento / decisão
TRACK_HISTORY = 12         # histórico por objeto rastreado
MIN_POSITIVES = 5          # quantos frames positivos no histórico para confirmar
ALERT_HOLD_SECONDS = 1.5   # mantém alerta por um tempo para não piscar
MAX_TRACK_AGE = 2.0        # remove tracks sumidos há muito tempo

# Visual
WINDOW_NAME = "Knife Detection"
SHOW_FPS = True
SHOW_DEBUG = True
DRAW_ALL_BOXES = True      # se False, desenha só caixas relevantes

# =========================
# INICIALIZAÇÃO
# =========================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)

# Tenta reduzir latência da câmera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("Nao foi possivel abrir a camera.")

# Histórico por track_id
track_memory = defaultdict(lambda: deque(maxlen=TRACK_HISTORY))
track_last_seen = {}

alert_until = 0.0
prev_time = time.time()


# =========================
# FUNÇÕES AUXILIARES
# =========================
def now():
    return time.time()


def cleanup_old_tracks(current_time: float):
    expired = [
        track_id
        for track_id, last_seen in track_last_seen.items()
        if (current_time - last_seen) > MAX_TRACK_AGE
    ]
    for track_id in expired:
        track_last_seen.pop(track_id, None)
        track_memory.pop(track_id, None)


def draw_banner(frame, text, color=(0, 0, 255)):
    cv2.rectangle(frame, (20, 15), (520, 85), color, -1)
    cv2.putText(
        frame,
        text,
        (35, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
        cv2.LINE_AA
    )


def draw_fps(frame, fps):
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


# =========================
# LOOP PRINCIPAL
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar frame.")
        break

    current_time = now()

    # Rastreamento persistente: melhor para vídeo do que tratar cada frame isoladamente
    results = model.track(
        source=frame,
        persist=True,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        device=DEVICE,
        half=USE_HALF,
        verbose=False,
        stream=False
    )

    annotated_frame = frame.copy()
    alert_active_this_frame = False
    relevant_boxes = []

    for r in results:
        boxes = r.boxes

        if boxes is None or len(boxes) == 0:
            continue

        # ids podem vir None em alguns casos
        track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
        classes = boxes.cls.int().cpu().tolist() if boxes.cls is not None else []
        confs = boxes.conf.cpu().tolist() if boxes.conf is not None else []
        xyxy_list = boxes.xyxy.int().cpu().tolist() if boxes.xyxy is not None else []

        for track_id, cls_id, conf, xyxy in zip(track_ids, classes, confs, xyxy_list):
            x1, y1, x2, y2 = xyxy

            is_knife = (cls_id == KNIFE_CLASS_ID) and (conf >= CONF_THRESHOLD)

            # Se não houver ID de track, usa um ID temporário baseado na caixa.
            # Não é o ideal, mas evita quebrar a lógica.
            if track_id is None:
                track_id = hash((x1 // 20, y1 // 20, x2 // 20, y2 // 20, cls_id))

            track_last_seen[track_id] = current_time
            track_memory[track_id].append(1 if is_knife else 0)

            positive_count = sum(track_memory[track_id])
            is_confirmed = positive_count >= MIN_POSITIVES

            if is_confirmed:
                alert_active_this_frame = True

            relevant_boxes.append({
                "track_id": track_id,
                "cls_id": cls_id,
                "conf": conf,
                "xyxy": (x1, y1, x2, y2),
                "positive_count": positive_count,
                "confirmed": is_confirmed,
                "is_knife": is_knife,
            })

    # Limpa tracks antigas
    cleanup_old_tracks(current_time)

    # Histerese do alerta: evita piscar
    if alert_active_this_frame:
        alert_until = current_time + ALERT_HOLD_SECONDS

    alert_on_screen = current_time < alert_until

    # Desenho
    for item in relevant_boxes:
        x1, y1, x2, y2 = item["xyxy"]
        conf = item["conf"]
        track_id = item["track_id"]
        confirmed = item["confirmed"]
        is_knife = item["is_knife"]
        positive_count = item["positive_count"]

        if not DRAW_ALL_BOXES and not is_knife:
            continue

        if confirmed:
            color = (0, 0, 255)   # vermelho forte
            label = f"KNIFE CONFIRMED | ID {track_id} | {conf:.2f} | {positive_count}/{TRACK_HISTORY}"
            thickness = 3
        elif is_knife:
            color = (0, 165, 255) # laranja
            label = f"KNIFE SUSPECT | ID {track_id} | {conf:.2f} | {positive_count}/{TRACK_HISTORY}"
            thickness = 2
        else:
            color = (0, 255, 0)   # verde
            label = f"OTHER | ID {track_id} | {conf:.2f}"
            thickness = 2

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            annotated_frame,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )

    # Banner de alerta
    if alert_on_screen:
        draw_banner(annotated_frame, "ALERTA: FACA DETECTADA", (0, 0, 255))

    # Debug/FPS
    current_fps = 1.0 / max(1e-6, (current_time - prev_time))
    prev_time = current_time

    if SHOW_FPS:
        draw_fps(annotated_frame, current_fps)

    if SHOW_DEBUG:
        cv2.putText(
            annotated_frame,
            f"Tracks ativas: {len(track_memory)}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

    cv2.imshow(WINDOW_NAME, annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()