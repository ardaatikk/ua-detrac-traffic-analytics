import argparse
import json
import math

from collections import Counter

from tqdm import tqdm

from src.config.paths import (
    YOLO_IMAGES_DIR,
    YOLO_LABELS_DIR,
    YOLO_DATASET_YAML,
    YOLO_CONVERSION_REPORT,
    YOLO_VALIDATION_REPORT,
)

from src.config.dataset import (
    VEHICLE_CLASSES,
    YOLO_SPLITS,
)

from src.config.settings import SHOW_PROGRESS

from src.data.convert_to_yolo import load_dataset_splits


def validate_label_file(label_path):
    class_counts = Counter()
    errors = []

    try:
        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

    except OSError as exc:
        return class_counts, [f"Cannot read label: {exc}"]

    for line_number, line in enumerate(lines, start=1):
        values = line.split()

        if len(values) != 5:
            errors.append(
                f"{label_path}:{line_number}: "
                "Expected 5 values."
            )
            continue

        try:
            class_id = int(values[0])
            x, y, width, height = map(
                float,
                values[1:],
            )

        except ValueError:
            errors.append(
                f"{label_path}:{line_number}: "
                "Invalid numeric values."
            )
            continue

        if class_id not in VEHICLE_CLASSES.values():
            errors.append(
                f"{label_path}:{line_number}: "
                f"Unknown class ID {class_id}."
            )
            continue

        coordinates = (x, y, width, height)

        if not all(math.isfinite(v) for v in coordinates):
            errors.append(
                f"{label_path}:{line_number}: "
                "Non-finite coordinates."
            )
            continue

        if not all(0 <= v <= 1 for v in coordinates):
            errors.append(
                f"{label_path}:{line_number}: "
                "Coordinates outside [0, 1]."
            )
            continue

        if width <= 0 or height <= 0:
            errors.append(
                f"{label_path}:{line_number}: "
                "Invalid box dimensions."
            )
            continue

        tolerance = 1e-5

        if (
            x - width / 2 < -tolerance
            or x + width / 2 > 1 + tolerance
            or y - height / 2 < -tolerance
            or y + height / 2 > 1 + tolerance
        ):
            errors.append(
                f"{label_path}:{line_number}: "
                "Bounding box exceeds image boundaries."
            )
            continue

        class_counts[class_id] += 1

    return class_counts, errors


def validate_split(split, sequences, expected):
    image_root = YOLO_IMAGES_DIR / split
    label_root = YOLO_LABELS_DIR / split

    statistics = {
        "sequences": len(sequences),
        "frames": 0,
        "annotations": 0,
        "class_counts": Counter(),
        "errors": [],
    }

    actual_image_sequences = {
        path.name
        for path in image_root.iterdir()
        if path.is_dir()
    } if image_root.is_dir() else set()

    actual_label_sequences = {
        path.name
        for path in label_root.iterdir()
        if path.is_dir()
    } if label_root.is_dir() else set()

    expected_sequences = set(sequences)

    if actual_image_sequences != expected_sequences:
        statistics["errors"].append(
            f"{split}: Image sequence directories mismatch."
        )

    if actual_label_sequences != expected_sequences:
        statistics["errors"].append(
            f"{split}: Label sequence directories mismatch."
        )

    iterator = tqdm(
        sequences,
        desc=f"Validating {split}",
        unit="sequence",
        disable=not SHOW_PROGRESS,
    )

    for sequence in iterator:
        image_dir = image_root / sequence
        label_dir = label_root / sequence

        if not image_dir.is_dir():
            statistics["errors"].append(
                f"Missing image directory: {image_dir}"
            )
            continue

        if not label_dir.is_dir():
            statistics["errors"].append(
                f"Missing label directory: {label_dir}"
            )
            continue

        images = sorted(image_dir.glob("*.jpg"))
        labels = sorted(label_dir.glob("*.txt"))

        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}

        if image_stems != label_stems:
            statistics["errors"].append(
                f"{split}/{sequence}: "
                "Image and label filenames mismatch."
            )

        for image_path in images:
            if (
                not image_path.is_symlink()
                or not image_path.is_file()
            ):
                statistics["errors"].append(
                    f"Invalid image symlink: {image_path}"
                )

        for label_path in labels:
            counts, errors = validate_label_file(
                label_path
            )

            statistics["class_counts"].update(counts)
            statistics["errors"].extend(errors)

        statistics["frames"] += len(images)

    statistics["annotations"] = sum(
        statistics["class_counts"].values()
    )

    expected_classes = {
        VEHICLE_CLASSES[name]: count
        for name, count in expected["class_counts"].items()
    }

    checks = {
        "sequences": expected["sequences"],
        "frames": expected["frames"],
        "annotations": expected["annotations"],
    }

    for key, expected_value in checks.items():
        if statistics[key] != expected_value:
            statistics["errors"].append(
                f"{split}: {key} mismatch. "
                f"Expected {expected_value}, "
                f"got {statistics[key]}."
            )

    if dict(statistics["class_counts"]) != expected_classes:
        statistics["errors"].append(
            f"{split}: Class distribution mismatch."
        )

    statistics["class_counts"] = dict(
        statistics["class_counts"]
    )

    return statistics


def main():
    parser = argparse.ArgumentParser(
        description="Validate prepared YOLO dataset."
    )

    parser.parse_args()

    if not YOLO_CONVERSION_REPORT.is_file():
        raise FileNotFoundError(
            "YOLO conversion report not found."
        )

    if not YOLO_DATASET_YAML.is_file():
        raise FileNotFoundError(
            "YOLO dataset.yaml not found."
        )

    with YOLO_CONVERSION_REPORT.open(
        encoding="utf-8"
    ) as file:
        conversion_report = json.load(file)

    splits = load_dataset_splits()

    report = {
        "status": "PASS",
        "splits": {},
        "total_frames": 0,
        "total_annotations": 0,
        "errors": [],
    }

    for split in YOLO_SPLITS:
        statistics = validate_split(
            split=split,
            sequences=splits[split],
            expected=conversion_report["splits"][split],
        )

        report["splits"][split] = statistics

        report["total_frames"] += statistics["frames"]
        report["total_annotations"] += statistics["annotations"]

        report["errors"].extend(statistics["errors"])

        print(
            f"{split.upper()}: "
            f"{statistics['frames']:,} frames, "
            f"{statistics['annotations']:,} annotations, "
            f"{len(statistics['errors'])} errors"
        )

    if (
        report["total_frames"]
        != conversion_report["total_frames"]
    ):
        report["errors"].append(
            "Total frame count mismatch."
        )

    if (
        report["total_annotations"]
        != conversion_report["total_annotations"]
    ):
        report["errors"].append(
            "Total annotation count mismatch."
        )

    if report["errors"]:
        report["status"] = "FAIL"

    YOLO_VALIDATION_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with YOLO_VALIDATION_REPORT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    print("\n" + "=" * 60)
    print("YOLO Dataset Validation")
    print("=" * 60)

    print(f"Status: {report['status']}")
    print(f"Frames: {report['total_frames']:,}")
    print(f"Annotations: {report['total_annotations']:,}")
    print(f"Errors: {len(report['errors'])}")
    print(f"Report: {YOLO_VALIDATION_REPORT}")

    if report["errors"]:
        print("\nFirst errors:")

        for error in report["errors"][:10]:
            print(f"- {error}")

        raise SystemExit(1)


if __name__ == "__main__":
    main()