from ultralytics import YOLO

def main():
    model = YOLO("yolov8m.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=842,
        batch=4,
        device=0,
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
        plots=True
    )

if __name__ == "__main__":
    main()