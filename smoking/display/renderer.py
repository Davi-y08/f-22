from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2


@dataclass(slots=True)
class DisplayFrame:
    camera_id: str
    window_name: str
    frame: Any
    max_width: int | None = None


class DisplayRenderer:
    def __init__(self) -> None:
        self._frames: dict[str, DisplayFrame] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._user_stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="display-renderer", daemon=True)

    @property
    def stop_requested(self) -> bool:
        return self._user_stop_event.is_set()

    def start(self) -> None:
        if self._thread.is_alive():
            return
        self._stop_event.clear()
        self._user_stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="display-renderer", daemon=True)
        self._thread.start()

    def submit(self, camera_id: str, window_name: str, frame: Any, max_width: int | None = None) -> None:
        with self._lock:
            self._frames[camera_id] = DisplayFrame(
                camera_id=camera_id,
                window_name=window_name,
                frame=frame,
                max_width=max_width,
            )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                snapshot = list(self._frames.values())

            if not snapshot:
                time.sleep(0.05)
                continue

            for item in snapshot:
                frame = _resize_if_needed(item.frame, item.max_width)
                cv2.imshow(item.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q"), ord("Q")}:
                self._user_stop_event.set()
                self._stop_event.set()
                break


def _resize_if_needed(frame: Any, max_width: int | None) -> Any:
    if not max_width:
        return frame

    height, width = frame.shape[:2]
    if width <= max_width:
        return frame

    scale = max_width / float(width)
    new_height = max(1, int(height * scale))
    return cv2.resize(frame, (max_width, new_height), interpolation=cv2.INTER_AREA)
