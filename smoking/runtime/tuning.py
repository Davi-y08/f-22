from __future__ import annotations

import os
from typing import Any

import cv2


def apply_runtime_tuning(
    enabled_camera_count: int,
    device_preference: str,
    logger: Any,
) -> None:
    camera_count = max(1, int(enabled_camera_count))
    cpu_count = os.cpu_count() or 4
    per_camera_budget = max(1, cpu_count // camera_count)
    opencv_threads = max(1, min(4, per_camera_budget))
    onnx_threads = max(1, min(6, per_camera_budget))

    try:
        cv2.setNumThreads(opencv_threads)
    except Exception:
        pass

    os.environ.setdefault("STEALTH_LENS_ONNX_THREADS", str(onnx_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, min(8, per_camera_budget))))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    _tune_torch_cpu_threads(
        device_preference=device_preference,
        per_camera_budget=per_camera_budget,
    )

    logger.info(
        "runtime_tuning_applied",
        extra={
            "enabled_cameras": camera_count,
            "cpu_count": cpu_count,
            "opencv_threads": opencv_threads,
            "onnx_threads": onnx_threads,
        },
    )


def _tune_torch_cpu_threads(device_preference: str, per_camera_budget: int) -> None:
    normalized_device = str(device_preference or "auto").strip().lower()
    if normalized_device not in {"cpu", "auto"}:
        return

    try:
        import torch  # type: ignore

        use_cpu_threads = normalized_device == "cpu" or not torch.cuda.is_available()
        if not use_cpu_threads:
            return

        torch_threads = max(1, min(8, per_camera_budget))
        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(max(1, min(2, torch_threads // 2)))
        except Exception:
            pass
    except Exception:
        pass
