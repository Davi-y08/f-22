from ultralytics import YOLO
import cv2
from collections import deque

model = YOLO("./runs/detect/train3/weights/best.pt")

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

history = deque(maxlen=10)

CONF_THRESHOLD = 0.6
ALERT_FRAMES = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    detected = False
    annotated_frame = frame.copy()

    for r in results:
        annotated_frame = r.plot()

        if r.boxes is not None:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if conf > CONF_THRESHOLD and cls == 1:  # knife
                    detected = True

    history.append(detected)

    if sum(history) >= ALERT_FRAMES:
        cv2.putText(
            annotated_frame,
            "ALERTA: FACA DETECTADA",
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