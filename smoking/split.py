from pathlib import Path
import argparse
import random
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Separa parte do dataset de treino para validação.")
    parser.add_argument("--dataset-dir", default="dataset", help="Diretório raiz do dataset YOLO.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Percentual movido para validação.")
    parser.add_argument("--seed", type=int, default=42, help="Seed usada para embaralhar imagens.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    train_images = dataset_dir / "train" / "images"
    train_labels = dataset_dir / "train" / "labels"
    valid_images = dataset_dir / "valid" / "images"
    valid_labels = dataset_dir / "valid" / "labels"

    if not train_images.exists():
        print(f"Diretório de imagens não encontrado: {train_images}")
        return 1

    valid_images.mkdir(parents=True, exist_ok=True)
    valid_labels.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = [path for path in train_images.iterdir() if path.suffix.lower() in image_exts]
    random.shuffle(all_images)

    val_ratio = max(0.0, min(1.0, args.val_ratio))
    val_count = int(len(all_images) * val_ratio)

    for image_path in all_images[:val_count]:
        label_path = train_labels / f"{image_path.stem}.txt"
        shutil.move(str(image_path), str(valid_images / image_path.name))

        if label_path.exists():
            shutil.move(str(label_path), str(valid_labels / label_path.name))
        else:
            print(f"Aviso: label não encontrada para {image_path.name}")

    print(f"Movidas {val_count} imagem(ns) para validação.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
