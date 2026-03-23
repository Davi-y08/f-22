from ultralytics import YOLO
import cv2
from collections import deque

# Carregar modelo
model = YOLO("./runs/detect/train3/weights/best.pt")

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)

# Buffer para estabilidade (últimos frames)
history = deque(maxlen=10)

CONF_THRESHOLD = 0.6
ALERT_FRAMES = 5  # precisa detectar em X frames seguidos

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inferência SEM stream (mais correto)
    results = model(frame, conf=CONF_THRESHOLD)

    detected = False

    for r in results:
        boxes = r.boxes

        if boxes is not None:
            for box in boxes:
                conf = float(box.conf[0])

                if conf > CONF_THRESHOLD:
                    detected = True

        annotated_frame = r.plot()

    # Adiciona no histórico
    history.append(detected)

    # Verifica consistência
    if history.count(True) >= ALERT_FRAMES:
        cv2.putText(
            annotated_frame,
            "ALERTA: POSSIVEL ARMA",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()