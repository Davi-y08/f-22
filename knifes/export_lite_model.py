from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO


def main() -> int:
    root = Path(__file__).resolve().parent
    source_model = root / "runs" / "detect" / "train" / "weights" / "best.pt"
    output_dir = root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_model.exists():
        raise FileNotFoundError(f"Modelo treinado nao encontrado: {source_model}")

    model = YOLO(str(source_model))
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=640,
            opset=12,
            simplify=True,
            dynamic=False,
        )
    )
    target_path = output_dir / "knife_monitor.onnx"
    if exported_path.resolve() != target_path.resolve():
        target_path.write_bytes(exported_path.read_bytes())
    print(f"Modelo Lite exportado em: {target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
