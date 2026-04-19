from __future__ import annotations

from pathlib import Path
import shutil
import sys


def main() -> int:
    source = Path("runs/detect/train/weights/best.pt").resolve()
    target = Path("models/smoking_monitor.onnx").resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        print(f"Modelo fonte não encontrado: {source}")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics não está instalado. Rode: pip install -r requirements.txt")
        return 1

    print(f"Exportando ONNX de '{source}'...")
    model = YOLO(str(source))
    exported = model.export(format="onnx", imgsz=640, dynamic=False, simplify=True, opset=12)
    exported_path = Path(str(exported)).resolve()

    if not exported_path.exists():
        print("Falha ao exportar o modelo ONNX.")
        return 1

    if exported_path != target:
        shutil.copy2(exported_path, target)

    print(f"Modelo Lite pronto em: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
