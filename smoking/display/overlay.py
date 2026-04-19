from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from models.loader import Detection


def render_monitor_frame(
    frame: Any,
    camera_name: str,
    detections: Sequence[Detection],
    zones: Sequence[tuple[str, list[tuple[int, int]]]],
    status_lines: Sequence[str],
    recent_events: Sequence[str],
) -> Any:
    annotated = frame.copy()

    for zone_name, points in zones:
        if len(points) < 3:
            continue
        cv2.polylines(
            annotated,
            [np.array(_to_point_array(points), dtype=np.int32)],
            isClosed=True,
            color=(0, 180, 255),
            thickness=2,
        )
        label_x, label_y = points[0]
        cv2.putText(
            annotated,
            zone_name,
            (label_x, max(20, label_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 180, 255),
            2,
        )

    for detection in detections:
        if not detection.display:
            continue

        x1, y1, x2, y2 = detection.bbox
        color = detection.overlay_color or _color_for_label(detection.label)
        label = detection.overlay_label or _default_overlay_label(detection)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 90), (8, 12, 18), -1)
    cv2.putText(
        annotated,
        camera_name,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
    )

    for index, line in enumerate(status_lines[:3]):
        cv2.putText(
            annotated,
            line,
            (16, 54 + (index * 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (160, 230, 255),
            1,
        )

    event_y = annotated.shape[0] - 20
    for line in reversed(list(recent_events)[-3:]):
        cv2.putText(
            annotated,
            line,
            (16, event_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 220, 120),
            2,
        )
        event_y -= 22

    return annotated


def _color_for_label(label: str) -> tuple[int, int, int]:
    key = label.lower()
    if "smoking" in key:
        return (0, 0, 255)
    if "cigarette" in key:
        return (0, 80, 255)
    if "smoke" in key:
        return (200, 200, 200)
    if "person" in key:
        return (0, 220, 120)
    if "knife" in key:
        return (0, 0, 255)
    return (255, 180, 0)


def _to_point_array(points: Sequence[tuple[int, int]]) -> list[list[int]]:
    return [[int(x), int(y)] for x, y in points]


def _default_overlay_label(detection: Detection) -> str:
    if detection.track_id is not None:
        return f"{detection.label} #{detection.track_id} {detection.confidence:.2f}"
    return f"{detection.label} {detection.confidence:.2f}"
