from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class DisplayFrame:
    camera_id: str
    window_name: str
    frame: Any
    max_width: int | None = None
    fullscreen: bool = False
    fit_mode: str = "contain"
    interpolation: str = "auto"
    enhance: bool = False


@dataclass(slots=True)
class _WindowState:
    camera_id: str
    window_name: str
    fullscreen_applied: bool = False


class DisplayRenderer:
    def __init__(self) -> None:
        self._frames: dict[str, DisplayFrame] = {}
        self._windows: dict[str, _WindowState] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._user_stop_event = threading.Event()
        self._manual_fullscreen_override: bool | None = None
        self._screen_size = _primary_screen_size()
        self._thread = threading.Thread(target=self._loop, name="display-renderer", daemon=True)

    @property
    def stop_requested(self) -> bool:
        return self._user_stop_event.is_set()

    def start(self) -> None:
        if self._thread.is_alive():
            return
        self._stop_event.clear()
        self._user_stop_event.clear()
        self._manual_fullscreen_override = None
        self._thread = threading.Thread(target=self._loop, name="display-renderer", daemon=True)
        self._thread.start()

    def submit(
        self,
        camera_id: str,
        window_name: str,
        frame: Any,
        max_width: int | None = None,
        fullscreen: bool = False,
        fit_mode: str = "contain",
        interpolation: str = "auto",
        enhance: bool = False,
    ) -> None:
        with self._lock:
            self._frames[camera_id] = DisplayFrame(
                camera_id=camera_id,
                window_name=window_name,
                frame=frame,
                max_width=max_width,
                fullscreen=fullscreen,
                fit_mode=fit_mode,
                interpolation=interpolation,
                enhance=enhance,
            )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self._windows.clear()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                snapshot = list(self._frames.values())

            if not snapshot:
                time.sleep(0.05)
                continue

            active_camera_ids = {item.camera_id for item in snapshot}
            self._destroy_stale_windows(active_camera_ids)

            for item in snapshot:
                effective_fullscreen = (
                    item.fullscreen
                    if self._manual_fullscreen_override is None
                    else self._manual_fullscreen_override
                )
                self._ensure_window(item.camera_id, item.window_name, effective_fullscreen)
                frame = _prepare_display_frame(
                    frame=item.frame,
                    max_width=item.max_width,
                    fullscreen=effective_fullscreen,
                    fit_mode=item.fit_mode,
                    interpolation=item.interpolation,
                    enhance=item.enhance,
                    screen_size=self._screen_size,
                )
                cv2.imshow(item.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q"), ord("Q")}:
                self._user_stop_event.set()
                self._stop_event.set()
                break
            if key in {ord("f"), ord("F")}:
                self._toggle_fullscreen_override()

    def _ensure_window(self, camera_id: str, window_name: str, fullscreen: bool) -> None:
        state = self._windows.get(camera_id)

        if state is not None and state.window_name != window_name:
            try:
                cv2.destroyWindow(state.window_name)
            except Exception:
                pass
            state = None

        if state is None:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            state = _WindowState(camera_id=camera_id, window_name=window_name)
            self._windows[camera_id] = state

        if state.fullscreen_applied != fullscreen:
            try:
                cv2.setWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
            except Exception:
                pass
            state.fullscreen_applied = fullscreen

    def _destroy_stale_windows(self, active_camera_ids: set[str]) -> None:
        stale = [camera_id for camera_id in self._windows if camera_id not in active_camera_ids]
        for camera_id in stale:
            state = self._windows.pop(camera_id)
            try:
                cv2.destroyWindow(state.window_name)
            except Exception:
                pass

    def _toggle_fullscreen_override(self) -> None:
        if self._manual_fullscreen_override is None:
            self._manual_fullscreen_override = True
            return
        if self._manual_fullscreen_override:
            self._manual_fullscreen_override = False
            return
        self._manual_fullscreen_override = None


def _prepare_display_frame(
    frame: Any,
    max_width: int | None,
    fullscreen: bool,
    fit_mode: str,
    interpolation: str,
    enhance: bool,
    screen_size: tuple[int, int],
) -> Any:
    output = frame
    if enhance:
        output = _enhance_frame(output)

    if fullscreen:
        screen_w, screen_h = screen_size
        return _fit_to_screen(
            frame=output,
            target_width=screen_w,
            target_height=screen_h,
            fit_mode=fit_mode,
            interpolation=interpolation,
        )

    return _resize_if_needed(output, max_width=max_width, interpolation=interpolation)


def _resize_if_needed(frame: Any, max_width: int | None, interpolation: str) -> Any:
    if not max_width or max_width <= 0:
        return frame

    height, width = frame.shape[:2]
    if width <= max_width:
        return frame

    scale = max_width / float(width)
    new_height = max(1, int(height * scale))
    inter = _choose_interpolation(interpolation, scale)
    return cv2.resize(frame, (max_width, new_height), interpolation=inter)


def _fit_to_screen(
    frame: Any,
    target_width: int,
    target_height: int,
    fit_mode: str,
    interpolation: str,
) -> Any:
    height, width = frame.shape[:2]
    target_width = max(320, int(target_width))
    target_height = max(240, int(target_height))

    if fit_mode == "stretch":
        inter = _choose_interpolation(interpolation, target_width / float(max(1, width)))
        return cv2.resize(frame, (target_width, target_height), interpolation=inter)

    if fit_mode == "cover":
        scale = max(target_width / float(width), target_height / float(height))
        resized_w = max(1, int(round(width * scale)))
        resized_h = max(1, int(round(height * scale)))
        inter = _choose_interpolation(interpolation, scale)
        resized = cv2.resize(frame, (resized_w, resized_h), interpolation=inter)

        x_start = max(0, (resized_w - target_width) // 2)
        y_start = max(0, (resized_h - target_height) // 2)
        x_end = x_start + target_width
        y_end = y_start + target_height
        return resized[y_start:y_end, x_start:x_end]

    # contain (default)
    scale = min(target_width / float(width), target_height / float(height))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    inter = _choose_interpolation(interpolation, scale)
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=inter)

    canvas = np.zeros((target_height, target_width, 3), dtype=resized.dtype)
    x_start = max(0, (target_width - resized_w) // 2)
    y_start = max(0, (target_height - resized_h) // 2)
    canvas[y_start : y_start + resized_h, x_start : x_start + resized_w] = resized
    return canvas


def _enhance_frame(frame: Any) -> Any:
    # Unsharp mask leve para melhorar nitidez percebida sem "estourar" o vídeo.
    blurred = cv2.GaussianBlur(frame, (0, 0), 1.05)
    return cv2.addWeighted(frame, 1.12, blurred, -0.12, 0)


def _choose_interpolation(mode: str, scale: float) -> int:
    normalized = str(mode or "auto").strip().lower()
    if normalized == "nearest":
        return cv2.INTER_NEAREST
    if normalized == "linear":
        return cv2.INTER_LINEAR
    if normalized == "area":
        return cv2.INTER_AREA
    if normalized == "cubic":
        return cv2.INTER_CUBIC
    if normalized == "lanczos":
        return cv2.INTER_LANCZOS4

    if scale >= 1.0:
        return cv2.INTER_CUBIC
    return cv2.INTER_AREA


def _primary_screen_size() -> tuple[int, int]:
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        try:
            user32.SetProcessDPIAware()  # type: ignore[attr-defined]
        except Exception:
            pass
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    return 1920, 1080
