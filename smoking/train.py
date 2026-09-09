from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o modelo de detecção de tabagismo.")
    parser.add_argument("--data", default="dataset/data.yaml", help="Caminho para o data.yaml.")
    parser.add_argument(
        "--base-model",
        default="yolov8n.pt",
        help="Modelo base do YOLO. O Ultralytics baixa automaticamente se necessário.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Quantidade de épocas.")
    parser.add_argument("--imgsz", type=int, default=842, help="Tamanho de imagem para treino.")
    parser.add_argument("--batch", type=int, default=4, help="Tamanho do batch.")
    parser.add_argument("--device", default="0", help="Dispositivo do treino. Use cpu ou índice da GPU.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = YOLO(args.base_model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=4,

        patience=20,
        close_mosaic=10,

        optimizer="auto",
        lr0=0.005,

        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5,
        translate=0.05,
        scale=0.3,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,

        amp=True,
        cache=False,
        pretrained=True,
        val=True,
        save=True,
        plots=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
