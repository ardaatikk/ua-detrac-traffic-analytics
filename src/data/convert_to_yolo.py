import argparse
import json
import os
import tempfile
import xml.etree.ElementTree as ET

from collections import Counter
from pathlib import Path

from tqdm import tqdm

from src.config.paths import (
    RAW_DATA_DIR,
    IMAGES_DIR,
    DATASET_SPLIT_FILE,
    YOLO_DATASET_DIR,
    YOLO_IMAGES_DIR,
    YOLO_LABELS_DIR,
    YOLO_DATASET_YAML,
    YOLO_CONVERSION_REPORT,
)

from src.config.dataset import (
    ANNOTATION_FILES,
    VEHICLE_CLASSES,
    YOLO_CLASS_NAMES,
    YOLO_SPLITS,
    FRAME_NUMBER_ATTRIBUTE,
    FRAME_FILENAME_PREFIX,
    FRAME_FILENAME_DIGITS,
    FRAME_FILENAME_EXTENSION,
    BOUNDING_BOX_ATTRIBUTES,
    VEHICLE_TYPE_ATTRIBUTE,
    EXPECTED_IMAGE_WIDTH,
    EXPECTED_IMAGE_HEIGHT,
)

from src.config.settings import (
    YOLO_LABEL_PRECISION,
    SHOW_PROGRESS,
)


def convert_bbox_to_yolo(
    left: float,
    top: float,
    width: float,
    height: float,
    image_width: int = EXPECTED_IMAGE_WIDTH,
    image_height: int = EXPECTED_IMAGE_HEIGHT,
) -> tuple[float, float, float, float]:
    """Convert an absolute XYWH box to normalized YOLO coordinates."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    if width <= 0 or height <= 0:
        raise ValueError("Bounding box dimensions must be positive.")

    # Original box boundaries
    x1 = left
    y1 = top
    x2 = left + width
    y2 = top + height

    # Clip to image boundaries
    x1 = max(0.0, min(x1, image_width))
    y1 = max(0.0, min(y1, image_height))
    x2 = max(0.0, min(x2, image_width))
    y2 = max(0.0, min(y2, image_height))

    # Recalculate dimensions after clipping
    clipped_width = x2 - x1
    clipped_height = y2 - y1

    if clipped_width <= 0 or clipped_height <= 0:
        raise ValueError(
            "Bounding box is outside image boundaries."
        )

    x_center = (x1 + x2) / 2 / image_width
    y_center = (y1 + y2) / 2 / image_height

    normalized_width = clipped_width / image_width
    normalized_height = clipped_height / image_height

    return (
        x_center,
        y_center,
        normalized_width,
        normalized_height,
    )
  
    
def load_dataset_splits():
    with DATASET_SPLIT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        split_data = json.load(file)

    train_sequences = set(split_data["train"])
    val_sequences = set(split_data["val"])

    train_annotation_dir = (
        RAW_DATA_DIR
        / ANNOTATION_FILES["train"]["directory"]
    )

    test_annotation_dir = (
        RAW_DATA_DIR
        / ANNOTATION_FILES["test"]["directory"]
    )

    available_train = {
        path.stem
        for path in train_annotation_dir.glob("*.xml")
    }

    available_test = {
        path.stem
        for path in test_annotation_dir.glob("*.xml")
    }

    if train_sequences & val_sequences:
        raise ValueError(
            "Train and validation sequences overlap."
        )

    if train_sequences | val_sequences != available_train:
        raise ValueError(
            "Dataset split does not match training XML files."
        )

    if available_test & (train_sequences | val_sequences):
        raise ValueError(
            "Test sequences overlap with train/validation."
        )

    return {
        "train": sorted(train_sequences),
        "val": sorted(val_sequences),
        "test": sorted(available_test),
    }
  
    
def ensure_image_symlink(source: Path, destination: Path):
    if not source.is_file():
        raise FileNotFoundError(
            f"Source image not found: {source}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = source.resolve()

    if destination.is_symlink():
        if destination.resolve() == source:
            return False

        destination.unlink()

    elif destination.exists():
        raise FileExistsError(
            f"Expected symlink, found regular file: {destination}"
        )

    relative_target = os.path.relpath(
        source,
        start=destination.parent,
    )

    destination.symlink_to(relative_target)

    return True


def build_frame_labels(frame):
    lines = []
    class_counts = Counter()
    rejected_boxes = 0

    for target in frame.findall("./target_list/target"):
        box = target.find("box")
        attribute = target.find("attribute")

        if box is None or attribute is None:
            rejected_boxes += 1
            continue

        vehicle_type = attribute.attrib.get(
            VEHICLE_TYPE_ATTRIBUTE
        )

        if vehicle_type not in VEHICLE_CLASSES:
            rejected_boxes += 1
            continue

        try:
            left, top, width, height = (
                float(box.attrib[name])
                for name in BOUNDING_BOX_ATTRIBUTES
            )

            x, y, w, h = convert_bbox_to_yolo(
                left=left,
                top=top,
                width=width,
                height=height,
                image_width=EXPECTED_IMAGE_WIDTH,
                image_height=EXPECTED_IMAGE_HEIGHT,
            )

        except (ValueError, KeyError):
            rejected_boxes += 1
            continue

        class_id = VEHICLE_CLASSES[vehicle_type]

        values = (x, y, w, h)

        coordinates = " ".join(
            f"{value:.{YOLO_LABEL_PRECISION}f}"
            for value in values
        )

        lines.append(f"{class_id} {coordinates}")

        class_counts[vehicle_type] += 1

    content = "\n".join(lines)

    if content:
        content += "\n"

    return content, class_counts, rejected_boxes


def write_label_if_changed(path: Path, content: str):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.is_file():
        existing_content = path.read_text(
            encoding="utf-8"
        )

        if existing_content == content:
            return False

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}_",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(content)
            temporary_path = Path(temporary_file.name)

        temporary_path.replace(path)

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()

    return True


def convert_sequence(
    sequence_name: str,
    split: str,
    annotation_dir: Path,
    progress_enabled: bool,
):
    xml_path = annotation_dir / f"{sequence_name}.xml"

    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)

    root = ET.parse(xml_path).getroot()

    if root.attrib.get("name") != sequence_name:
        raise ValueError(
            f"Sequence name mismatch: {xml_path}"
        )

    frames = root.findall("frame")

    statistics = {
        "sequence": sequence_name,
        "frames": 0,
        "annotations": 0,
        "rejected_boxes": 0,
        "created_links": 0,
        "written_labels": 0,
        "class_counts": Counter(),
    }

    image_source_dir = IMAGES_DIR / sequence_name

    image_destination_dir = (
        YOLO_IMAGES_DIR / split / sequence_name
    )

    label_destination_dir = (
        YOLO_LABELS_DIR / split / sequence_name
    )

    iterator = frames

    seen_frame_numbers = set()

    for frame in iterator:
        frame_number = int(
            frame.attrib[FRAME_NUMBER_ATTRIBUTE]
        )

        if frame_number in seen_frame_numbers:
            raise ValueError(
                f"Duplicate frame {frame_number} "
                f"in {sequence_name}"
            )

        seen_frame_numbers.add(frame_number)

        filename = (
            f"{FRAME_FILENAME_PREFIX}"
            f"{frame_number:0{FRAME_FILENAME_DIGITS}d}"
            f"{FRAME_FILENAME_EXTENSION}"
        )

        image_source = image_source_dir / filename

        image_destination = (
            image_destination_dir / filename
        )

        label_destination = (
            label_destination_dir
            / f"{Path(filename).stem}.txt"
        )

        label_content, class_counts, rejected = (
            build_frame_labels(frame)
        )

        if ensure_image_symlink(
            image_source,
            image_destination,
        ):
            statistics["created_links"] += 1

        if write_label_if_changed(
            label_destination,
            label_content,
        ):
            statistics["written_labels"] += 1

        statistics["frames"] += 1
        statistics["annotations"] += sum(
            class_counts.values()
        )

        statistics["rejected_boxes"] += rejected
        statistics["class_counts"].update(class_counts)

    statistics["class_counts"] = dict(
        statistics["class_counts"]
    )

    return statistics


def write_dataset_yaml():
    final_dataset_dir = Path(
        os.environ.get(
            "UA_DETRAC_YOLO_FINAL_DIR",
            str(YOLO_DATASET_DIR),
        )
    ).resolve()

    lines = [
        f"path: {json.dumps(str(final_dataset_dir))}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]

    for class_id, class_name in enumerate(YOLO_CLASS_NAMES):
        lines.append(f"  {class_id}: {class_name}")

    content = "\n".join(lines) + "\n"

    write_label_if_changed(
        YOLO_DATASET_YAML,
        content,
    )
    
    
def main():
    parser = argparse.ArgumentParser(
        description="Convert UA-DETRAC annotations to YOLO format."
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )

    args = parser.parse_args()

    progress_enabled = (
        SHOW_PROGRESS and not args.no_progress
    )

    splits = load_dataset_splits()

    print("=" * 60)
    print("UA-DETRAC | XML to YOLO Conversion")
    print("=" * 60)

    report = {
        "splits": {},
        "total_frames": 0,
        "total_annotations": 0,
        "total_rejected_boxes": 0,
    }

    for split in YOLO_SPLITS:
        sequences = splits[split]

        annotation_source = (
            "test" if split == "test" else "train"
        )

        annotation_dir = (
            RAW_DATA_DIR
            / ANNOTATION_FILES[annotation_source]["directory"]
        )

        print(
            f"\nProcessing {split}: "
            f"{len(sequences)} sequences"
        )

        split_statistics = {
            "sequences": len(sequences),
            "frames": 0,
            "annotations": 0,
            "rejected_boxes": 0,
            "created_links": 0,
            "written_labels": 0,
            "class_counts": Counter(),
        }

        iterator = tqdm(
            sequences,
            desc=f"Converting {split}",
            unit="sequence",
            position=0,
            leave=True,
            disable=not progress_enabled,
        )

        for sequence_name in iterator:
            sequence_statistics = convert_sequence(
                sequence_name=sequence_name,
                split=split,
                annotation_dir=annotation_dir,
                progress_enabled=progress_enabled,
            )

            for key in (
                "frames",
                "annotations",
                "rejected_boxes",
                "created_links",
                "written_labels",
            ):
                split_statistics[key] += (
                    sequence_statistics[key]
                )

            split_statistics["class_counts"].update(
                sequence_statistics["class_counts"]
            )

        split_statistics["class_counts"] = dict(
            split_statistics["class_counts"]
        )

        report["splits"][split] = split_statistics

        report["total_frames"] += (
            split_statistics["frames"]
        )

        report["total_annotations"] += (
            split_statistics["annotations"]
        )

        report["total_rejected_boxes"] += (
            split_statistics["rejected_boxes"]
        )

        print(
            f"Completed {split}: "
            f"{split_statistics['frames']:,} frames, "
            f"{split_statistics['annotations']:,} annotations"
        )

    YOLO_CONVERSION_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with YOLO_CONVERSION_REPORT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    if report["total_rejected_boxes"]:
        raise ValueError(
            "Conversion encountered "
            f"{report['total_rejected_boxes']} "
            "invalid annotations. "
            f"See {YOLO_CONVERSION_REPORT}"
        )

    write_dataset_yaml()

    print("\n" + "=" * 60)
    print("YOLO Conversion Completed")
    print("=" * 60)

    print(
        f"Frames: {report['total_frames']:,}"
    )

    print(
        f"Annotations: {report['total_annotations']:,}"
    )

    print(
        f"Rejected boxes: {report['total_rejected_boxes']:,}"
    )

    print(f"Dataset YAML: {YOLO_DATASET_YAML}")
    print(f"Report: {YOLO_CONVERSION_REPORT}")


if __name__ == "__main__":
    main()