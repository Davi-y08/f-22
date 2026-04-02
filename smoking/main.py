from collections import defaultdict
from ultralytics import YOLO
import cv2
import math


MODEL_PATH = "runs/detect/train/weights/best.pt"
VIDEO_SOURCE = 0 

CONFIDENCE = 0.25
IOU = 0.45

PERSON_CLASS_NAME = "person"
CIGARETTE_CLASS_NAME = "cigarette"

MAX_DISTANCE_TO_PERSON = 80

MIN_FRAMES_SMOKING = 15


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def distance_point_to_box(px, py, box):
    x1, y1, x2, y2 = box

    closest_x = max(x1, min(px, x2))
    closest_y = max(y1, min(py, y2))

    return math.hypot(px - closest_x, py - closest_y)


def point_inside_box(px, py, box):
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def expand_box(box, margin):
    x1, y1, x2, y2 = box
    return (x1 - margin, y1 - margin, x2 + margin, y2 + margin)


def main():
    model = YOLO(MODEL_PATH)

    class_names = model.names
    person_class_id = None
    cigarette_class_id = None

    for class_id, name in class_names.items():
        if name == PERSON_CLASS_NAME:
            person_class_id = class_id
        elif name == CIGARETTE_CLASS_NAME:
            cigarette_class_id = class_id

    if person_class_id is None:
        raise ValueError(f"Classe '{PERSON_CLASS_NAME}' não encontrada no modelo.")

    if cigarette_class_id is None:
        raise ValueError(f"Classe '{CIGARETTE_CLASS_NAME}' não encontrada no modelo.")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a câmera/vídeo.")

    smoking_counter = defaultdict(int)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.track(
            source=frame,
            persist=True,
            conf=CONFIDENCE,
            iou=IOU,
            verbose=False
        )

        result = results[0]

        persons = []
        cigarettes = []

        if result.boxes is not None and result.boxes.xyxy is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            ids = []

            if result.boxes.id is not None:
                ids = result.boxes.id.int().cpu().tolist()
            else:
                ids = [-1] * len(boxes_xyxy)

            for box, cls_id, track_id in zip(boxes_xyxy, classes, ids):
                cls_id = int(cls_id)

                if cls_id == person_class_id:
                    persons.append({
                        "box": box,
                        "track_id": int(track_id)
                    })
                elif cls_id == cigarette_class_id:
                    cigarettes.append({
                        "box": box
                    })

        current_frame_person_ids = set()

        for person in persons:
            person_box = person["box"]
            track_id = person["track_id"]
            current_frame_person_ids.add(track_id)

            x1, y1, x2, y2 = map(int, person_box)
            expanded_person_box = expand_box(person_box, MAX_DISTANCE_TO_PERSON)

            near_cigarette = False
            matched_cigarette_box = None

            for cigarette in cigarettes:
                cig_box = cigarette["box"]
                cig_cx, cig_cy = box_center(cig_box)

                if point_inside_box(cig_cx, cig_cy, expanded_person_box):
                    near_cigarette = True
                    matched_cigarette_box = cig_box
                    break

                dist = distance_point_to_box(cig_cx, cig_cy, person_box)
                if dist <= MAX_DISTANCE_TO_PERSON:
                    near_cigarette = True
                    matched_cigarette_box = cig_box
                    break

            if near_cigarette:
                smoking_counter[track_id] += 1
            else:
                smoking_counter[track_id] = max(0, smoking_counter[track_id] - 1)

            is_smoking = smoking_counter[track_id] >= MIN_FRAMES_SMOKING

            if is_smoking:
                color = (0, 0, 255)
                label = f"SMOKING ID {track_id}"
            else:
                color = (0, 255, 0)
                label = f"Person ID {track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            cv2.putText(
                frame,
                f"frames={smoking_counter[track_id]}",
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            if matched_cigarette_box is not None:
                cx1, cy1, cx2, cy2 = map(int, matched_cigarette_box)
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 0), 2)
                cv2.putText(
                    frame,
                    "cigarette",
                    (cx1, max(20, cy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

        ids_to_remove = []
        for track_id in smoking_counter:
            if track_id not in current_frame_person_ids and smoking_counter[track_id] == 0:
                ids_to_remove.append(track_id)

        for track_id in ids_to_remove:
            del smoking_counter[track_id]

        cv2.imshow("Smoking Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()