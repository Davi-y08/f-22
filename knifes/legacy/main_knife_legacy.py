import cv2
import time
import os
import threading
from collections import deque, defaultdict
from datetime import datetime
from ultralytics import YOLO

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


# ==========================================
# CONFIGURAÇÕES
# ==========================================
MODEL_PATH = "./runs/detect/train3/weights/best.pt"
CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Classe da faca no seu dataset
KNIFE_CLASS_ID = 1

# Inferência
CONF_THRESHOLD = 0.60
IOU_THRESHOLD = 0.45
IMG_SIZE = 640
DEVICE = 0              # 0 = GPU, "cpu" = CPU
USE_HALF = True         # use False se der erro
USE_STREAM_BUFFER = False

# Lógica temporal
TRACK_HISTORY = 12
MIN_POSITIVES = 5
ALERT_HOLD_SECONDS = 1.5
ALERT_COOLDOWN_SECONDS = 4.0
MAX_TRACK_AGE = 2.0

# Gravação
SAVE_DIR = "detections"
SCREENSHOT_DIR = os.path.join(SAVE_DIR, "screenshots")
VIDEO_DIR = os.path.join(SAVE_DIR, "videos")
RECORD_SECONDS_AFTER_ALERT = 8
VIDEO_FPS = 20.0

# Visual
WINDOW_NAME = "Knife Detection Advanced"
SHOW_FPS = True
SHOW_DEBUG = True
DRAW_NON_KNIFE = False

# Áudio
ENABLE_BEEP = True
BEEP_FREQ = 1800
BEEP_DURATION_MS = 350


# ==========================================
# PREPARAÇÃO DE PASTAS
# ==========================================
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)


# ==========================================
# CAPTURA EM THREAD
# ==========================================
class CameraStream:
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


# ==========================================
# UTILITÁRIOS
# ==========================================
def now():
    return time.time()


def timestamp_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def draw_banner(frame, text, color=(0, 0, 255)):
    cv2.rectangle(frame, (20, 15), (620, 85), color, -1)
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


def draw_text(frame, text, y, color=(255, 255, 0)):
    cv2.putText(
        frame,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA
    )


def play_alert_beep():
    if ENABLE_BEEP and HAS_WINSOUND:
        try:
            winsound.Beep(BEEP_FREQ, BEEP_DURATION_MS)
        except Exception:
            pass


def create_video_writer(frame_width, frame_height):
    filename = os.path.join(VIDEO_DIR, f"alert_{timestamp_str()}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filename, fourcc, VIDEO_FPS, (frame_width, frame_height))
    return writer, filename


def save_screenshot(frame):
    filename = os.path.join(SCREENSHOT_DIR, f"evidence_{timestamp_str()}.jpg")
    cv2.imwrite(filename, frame)
    return filename


def cleanup_old_tracks(track_memory, track_last_seen, current_time):
    expired = [
        track_id
        for track_id, last_seen in track_last_seen.items()
        if current_time - last_seen > MAX_TRACK_AGE
    ]
    for track_id in expired:
        track_last_seen.pop(track_id, None)
        track_memory.pop(track_id, None)


# ==========================================
# MAIN
# ==========================================
def main():
    model = YOLO(MODEL_PATH)

    stream = CameraStream(
        src=CAMERA_INDEX,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT
    )

    track_memory = defaultdict(lambda: deque(maxlen=TRACK_HISTORY))
    track_last_seen = {}

    alert_until = 0.0
    last_alert_time = 0.0
    evidence_saved_for_current_alert = False

    video_writer = None
    video_filename = None
    record_until = 0.0

    prev_time = now()

    print("Sistema iniciado. Pressione Q para sair.")

    while True:
        ret, frame = stream.read()
        if not ret or frame is None:
            continue

        current_time = now()
        annotated_frame = frame.copy()

        # Inferência com tracking
        results = model.track(
            source=frame,
            persist=True,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            device=DEVICE,
            half=USE_HALF,
            verbose=False,
            stream=USE_STREAM_BUFFER
        )

        detected_confirmed = False
        detections_info = []

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
            classes = boxes.cls.int().cpu().tolist() if boxes.cls is not None else []
            confs = boxes.conf.cpu().tolist() if boxes.conf is not None else []
            xyxy_list = boxes.xyxy.int().cpu().tolist() if boxes.xyxy is not None else []

            for track_id, cls_id, conf, xyxy in zip(track_ids, classes, confs, xyxy_list):
                x1, y1, x2, y2 = xyxy
                is_knife = cls_id == KNIFE_CLASS_ID and conf >= CONF_THRESHOLD

                if track_id is None:
                    track_id = hash((x1 // 20, y1 // 20, x2 // 20, y2 // 20, cls_id))

                track_last_seen[track_id] = current_time
                track_memory[track_id].append(1 if is_knife else 0)

                positives = sum(track_memory[track_id])
                confirmed = positives >= MIN_POSITIVES

                if confirmed:
                    detected_confirmed = True

                detections_info.append({
                    "track_id": track_id,
                    "cls_id": cls_id,
                    "conf": conf,
                    "xyxy": (x1, y1, x2, y2),
                    "is_knife": is_knife,
                    "positives": positives,
                    "confirmed": confirmed
                })

        cleanup_old_tracks(track_memory, track_last_seen, current_time)

        # Acionamento do alerta com cooldown
        just_triggered = False
        if detected_confirmed:
            alert_until = current_time + ALERT_HOLD_SECONDS

            if (current_time - last_alert_time) >= ALERT_COOLDOWN_SECONDS:
                last_alert_time = current_time
                just_triggered = True
                evidence_saved_for_current_alert = False
                record_until = current_time + RECORD_SECONDS_AFTER_ALERT

        alert_active = current_time < alert_until

        # Desenhar caixas
        for det in detections_info:
            x1, y1, x2, y2 = det["xyxy"]
            conf = det["conf"]
            track_id = det["track_id"]
            is_knife = det["is_knife"]
            confirmed = det["confirmed"]
            positives = det["positives"]

            if confirmed:
                color = (0, 0, 255)
                thickness = 3
                label = f"KNIFE CONFIRMED | ID {track_id} | {conf:.2f} | {positives}/{TRACK_HISTORY}"
            elif is_knife:
                color = (0, 165, 255)
                thickness = 2
                label = f"KNIFE SUSPECT | ID {track_id} | {conf:.2f} | {positives}/{TRACK_HISTORY}"
            else:
                if not DRAW_NON_KNIFE:
                    continue
                color = (0, 255, 0)
                thickness = 2
                label = f"OTHER | ID {track_id} | {conf:.2f}"

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                annotated_frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA
            )

        # Banner de alerta
        if alert_active:
            draw_banner(annotated_frame, "ALERTA: FACA DETECTADA", (0, 0, 255))

        # Ao disparar: beep + screenshot + iniciar gravação
        if just_triggered:
            play_alert_beep()

            if not evidence_saved_for_current_alert:
                screenshot_file = save_screenshot(annotated_frame)
                print(f"[ALERTA] Screenshot salva em: {screenshot_file}")
                evidence_saved_for_current_alert = True

            if video_writer is None:
                h, w = annotated_frame.shape[:2]
                video_writer, video_filename = create_video_writer(w, h)
                print(f"[ALERTA] Gravacao iniciada: {video_filename}")

        # Continuar gravação enquanto estiver dentro da janela
        if video_writer is not None:
            if current_time <= record_until:
                video_writer.write(annotated_frame)
            else:
                video_writer.release()
                print(f"[INFO] Gravacao finalizada: {video_filename}")
                video_writer = None
                video_filename = None

        # FPS
        fps = 1.0 / max(1e-6, current_time - prev_time)
        prev_time = current_time

        if SHOW_FPS:
            draw_fps(annotated_frame, fps)

        if SHOW_DEBUG:
            draw_text(annotated_frame, f"Tracks ativas: {len(track_memory)}", 110)
            draw_text(annotated_frame, f"Cooldown restante: {max(0.0, ALERT_COOLDOWN_SECONDS - (current_time - last_alert_time)):.1f}s", 140)
            draw_text(annotated_frame, f"Gravando: {'SIM' if video_writer is not None else 'NAO'}", 170)

        cv2.imshow(WINDOW_NAME, annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    if video_writer is not None:
        video_writer.release()

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()