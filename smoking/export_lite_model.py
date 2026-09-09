from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporta o modelo treinado para ONNX Lite.")
    parser.add_argument(
        "--source",
        default="runs/detect/train/weights/best.pt",
        help="Modelo .pt treinado que será exportado.",
    )
    parser.add_argument(
        "--target",
        default="models/smoking_monitor.onnx",
        help="Destino do modelo ONNX usado pelo perfil Lite.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Tamanho de entrada usado na exportação.")
    parser.add_argument("--opset", type=int, default=12, help="Versão ONNX opset usada na exportação.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source).resolve()
    target = Path(args.target).resolve()
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
    exported = model.export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=True, opset=args.opset)
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
