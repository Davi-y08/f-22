from ultralytics import YOLO

def main():
    # Modelo mais forte (muito mais confiável que o n)
    model = YOLO("yolov8m.pt")

    model.train(
        data="dataset/data.yaml",

        # Treinamento mais longo
        epochs=120,

        # Resolução maior ajuda em objetos pequenos (facas principalmente)
        imgsz=640,

        # GPU
        device=0,

        # Melhor uso de CPU
        workers=4,

        # Batch maior se sua GPU aguentar
        batch=16,

        # Early stopping (evita overfitting)
        patience=20,

        # Data augmentation (CRÍTICO)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,

        # Salvar melhor modelo
        save=True
    )

if __name__ == "__main__":
    main()