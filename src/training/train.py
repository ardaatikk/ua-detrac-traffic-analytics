import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

from src.config.paths import (
    ARTIFACTS_DIR,
    YOLO_DATASET_YAML,
)
from src.config.settings import (
    BATCH_SIZE,
    EPOCHS,
    IMAGE_SIZE,
    RANDOM_SEED,
)


DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TRAINING_DIR = ARTIFACTS_DIR / "training"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train YOLO on UA-DETRAC."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=IMAGE_SIZE,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to use (0 < fraction <= 1).",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
    )

    parser.add_argument("--smoke", action="store_true",
                        help="Use the prepared sequence-aware smoke dataset.")
    return parser.parse_args(argv)


def main():
    args = parse_args()

    if not 0 < args.fraction <= 1:
        raise ValueError(
            "--fraction must be greater than 0 and at most 1."
        )

    if not YOLO_DATASET_YAML.exists():
        raise FileNotFoundError(
            f"YOLO dataset configuration not found: "
            f"{YOLO_DATASET_YAML}"
        )

    dataset_yaml = YOLO_DATASET_YAML
    if args.smoke:
        if args.fraction != 1.0:
            raise ValueError("--smoke requires --fraction 1.0 to preserve class coverage.")
        from src.config.paths import YOLO_SMOKE_DIR
        dataset_yaml = YOLO_SMOKE_DIR / "dataset.yaml"
        if not dataset_yaml.is_file():
            raise FileNotFoundError(
                "Prepare smoke data first: python -m src.data.create_smoke_dataset"
            )

    experiment_name = args.name or (
        f"yolo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    experiment_dir = TRAINING_DIR / experiment_name

    if experiment_dir.exists():
        raise FileExistsError(
            f"Experiment already exists: {experiment_dir}"
        )

    configuration = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "fraction": args.fraction,
        "patience": args.patience,
        "seed": RANDOM_SEED,
        "dataset": str(dataset_yaml),
        "smoke": args.smoke,
    }

    print("\nYOLO Training Configuration")
    print(json.dumps(configuration, indent=2))

    model = YOLO(args.model)

    model.train(
        data=str(dataset_yaml),
        deterministic=args.device != "mps",
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        fraction=args.fraction,
        patience=args.patience,
        seed=RANDOM_SEED,
        project=str(TRAINING_DIR),
        name=experiment_name,
        exist_ok=False,
        plots=True,
        save=True,
        verbose=True,
    )

    experiment_dir.mkdir(parents=True, exist_ok=True)

    config_path = experiment_dir / "training_config.json"

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=2)

    print(f"\nTraining completed: {experiment_dir}")
    print(f"Best weights: {experiment_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()