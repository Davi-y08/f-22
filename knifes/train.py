from ultralytics import YOLO

def main():
    model = YOLO("yolov8l.pt") 

    model.train(
        data="dataset/dataset.yaml",

        epochs=200,
        imgsz=640,          
        batch=8,            

        device=0,
        workers=4,
        cache=True,
        amp=True,

        cos_lr=True,
        patience=30,
        close_mosaic=10,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        mosaic=1.0,

        dropout=0.1,

        val=True,
        plots=False,
        save=True
    )

if __name__ == "__main__":
    main()