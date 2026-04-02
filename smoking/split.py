from pathlib import Path
import random
import shutil

random.seed(42)

dataset_dir = Path("dataset")
train_images = dataset_dir / "train" / "images"
train_labels = dataset_dir / "train" / "labels"

valid_images = dataset_dir / "valid" / "images"
valid_labels = dataset_dir / "valid" / "labels"

valid_images.mkdir(parents=True, exist_ok=True)
valid_labels.mkdir(parents=True, exist_ok=True)

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

all_images = [p for p in train_images.iterdir() if p.suffix.lower() in image_exts]
random.shuffle(all_images)

val_ratio = 0.2
val_count = int(len(all_images) * val_ratio)

val_images = all_images[:val_count]

for img_path in val_images:
    label_path = train_labels / f"{img_path.stem}.txt"

    shutil.move(str(img_path), str(valid_images / img_path.name))

    if label_path.exists():
        shutil.move(str(label_path), str(valid_labels / label_path.name))
    else:
        print(f"Aviso: label não encontrada para {img_path.name}")