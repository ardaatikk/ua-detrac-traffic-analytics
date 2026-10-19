from collections import Counter
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

from src.config.settings import SHOW_PROGRESS

from src.config.paths import (
    RAW_DATA_DIR,
    IMAGES_DIR,
    DATASET_VALIDATION_REPORT,
)

from src.config.dataset import (
    ANNOTATION_FILES,
    VEHICLE_CLASSES,
    FRAME_NUMBER_ATTRIBUTE,
    TARGET_ID_ATTRIBUTE,
    VEHICLE_TYPE_ATTRIBUTE,
    FRAME_FILENAME_PREFIX,
    FRAME_FILENAME_DIGITS,
    FRAME_FILENAME_EXTENSION,
    BOUNDING_BOX_ATTRIBUTES,
    IMAGE_EXTENSIONS,
    ALLOW_UNANNOTATED_IMAGES,
    REPORT_UNANNOTATED_SEQUENCES,
)


def build_frame_path(
    sequence_name: str,
    frame_number: int,
) -> Path:
    filename = (
        f"{FRAME_FILENAME_PREFIX}"
        f"{frame_number:0{FRAME_FILENAME_DIGITS}d}"
        f"{FRAME_FILENAME_EXTENSION}"
    )

    return IMAGES_DIR / sequence_name / filename


def validate_box(box: ET.Element) -> bool:
    try:
        coordinates = {
            attribute: float(box.attrib[attribute])
            for attribute in BOUNDING_BOX_ATTRIBUTES
        }

        return (
            coordinates["left"] >= 0
            and coordinates["top"] >= 0
            and coordinates["width"] > 0
            and coordinates["height"] > 0
        )

    except (KeyError, ValueError, TypeError):
        return False


def validate_sequence(
    xml_path: Path,
    statistics: dict,
    errors: list[str],
) -> None:
    try:
        root = ET.parse(xml_path).getroot()

    except ET.ParseError as exc:
        errors.append(
            f"Invalid XML: {xml_path.name}: {exc}"
        )
        return

    sequence_name = root.attrib.get("name")

    if not sequence_name:
        errors.append(
            f"Missing sequence name: {xml_path.name}"
        )
        return

    if sequence_name != xml_path.stem:
        errors.append(
            f"Sequence name mismatch: {xml_path.name}"
        )

    sequence_dir = IMAGES_DIR / sequence_name

    if not sequence_dir.is_dir():
        errors.append(
            f"Missing image sequence: {sequence_name}"
        )
        return

    statistics["sequences"] += 1

    tracking_ids = set()

    for frame in root.findall("frame"):
        statistics["frames"] += 1

        try:
            frame_number = int(
                frame.attrib[FRAME_NUMBER_ATTRIBUTE]
            )

        except (KeyError, ValueError):
            errors.append(
                f"Invalid frame number: {sequence_name}"
            )
            continue

        frame_path = build_frame_path(
            sequence_name,
            frame_number,
        )

        if not frame_path.is_file():
            errors.append(
                f"Missing frame: {frame_path}"
            )

        target_list = frame.find("target_list")

        if target_list is None:
            continue

        frame_target_ids = set()

        for target in target_list.findall("target"):
            statistics["annotations"] += 1

            target_id = target.attrib.get(
                TARGET_ID_ATTRIBUTE
            )

            if target_id is None:
                errors.append(
                    f"Missing target ID: "
                    f"{sequence_name}, frame {frame_number}"
                )

            else:
                if target_id in frame_target_ids:
                    errors.append(
                        f"Duplicate target ID: "
                        f"{sequence_name}, "
                        f"frame {frame_number}, "
                        f"target {target_id}"
                    )

                frame_target_ids.add(target_id)
                tracking_ids.add(target_id)

            box = target.find("box")

            if box is None or not validate_box(box):
                statistics["invalid_boxes"] += 1

                errors.append(
                    f"Invalid box: "
                    f"{sequence_name}, "
                    f"frame {frame_number}"
                )

            attribute = target.find("attribute")

            if attribute is None:
                statistics["missing_attributes"] += 1
                continue

            vehicle_type = attribute.attrib.get(
                VEHICLE_TYPE_ATTRIBUTE
            )

            if vehicle_type is None:
                statistics["missing_attributes"] += 1

            elif vehicle_type not in VEHICLE_CLASSES:
                statistics["unknown_classes"][
                    vehicle_type
                ] += 1

                errors.append(
                    f"Unknown vehicle class: {vehicle_type}"
                )

            else:
                statistics["class_distribution"][
                    vehicle_type
                ] += 1

            if target.find("occlusion") is not None:
                statistics["occluded_annotations"] += 1

    statistics["unique_vehicles"] += len(
        tracking_ids
    )

def analyze_frame_coverage(
    split_sequences: dict[str, set[str]],
) -> dict:
    coverage = {
        "total_images": 0,
        "annotated_frames": 0,
        "unannotated_images": 0,
        "splits": {},
        "sequences_with_gaps": {},
    }

    for split, sequences in split_sequences.items():
        split_images = 0
        split_annotated = 0
        split_unannotated = 0

        config = ANNOTATION_FILES[split]
        annotation_dir = RAW_DATA_DIR / config["directory"]

        for sequence_name in tqdm(
            sorted(sequences),
            desc=f"Checking coverage ({split})",
            unit="sequence",
            dynamic_ncols=True,
            position=0,
            leave=False,
            disable=not SHOW_PROGRESS,
        ):
            sequence_dir = IMAGES_DIR / sequence_name
            xml_path = annotation_dir / f"{sequence_name}.xml"

            if not sequence_dir.is_dir() or not xml_path.is_file():
                continue

            root = ET.parse(xml_path).getroot()

            image_names = {
                path.name
                for path in sequence_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            }

            annotated_names = {
                build_frame_path(
                    sequence_name,
                    int(frame.attrib[FRAME_NUMBER_ATTRIBUTE]),
                ).name
                for frame in root.findall("frame")
            }

            unannotated_names = image_names - annotated_names

            split_images += len(image_names)
            split_annotated += len(annotated_names)
            split_unannotated += len(unannotated_names)

            if unannotated_names and REPORT_UNANNOTATED_SEQUENCES:
                coverage["sequences_with_gaps"][
                    sequence_name
                ] = len(unannotated_names)

        coverage["splits"][split] = {
            "images": split_images,
            "annotated_frames": split_annotated,
            "unannotated_images": split_unannotated,
        }

        coverage["total_images"] += split_images
        coverage["annotated_frames"] += split_annotated
        coverage["unannotated_images"] += split_unannotated

    return coverage

def validate_dataset() -> dict:
    statistics = {
        "sequences": 0,
        "frames": 0,
        "annotations": 0,
        "unique_vehicles": 0,
        "invalid_boxes": 0,
        "missing_attributes": 0,
        "occluded_annotations": 0,
        "class_distribution": Counter(),
        "unknown_classes": Counter(),
    }

    errors = []
    split_sequences = {}

    for split, config in ANNOTATION_FILES.items():
        annotation_dir = (
            RAW_DATA_DIR / config["directory"]
        )

        xml_files = sorted(
            annotation_dir.glob("*.xml")
        )

        split_sequences[split] = {
            path.stem
            for path in xml_files
        }

        if len(xml_files) != config["expected_count"]:
            errors.append(
                f"Unexpected XML count for {split}: "
                f"{len(xml_files)}"
            )

        for xml_path in tqdm(
            xml_files,
            desc=f"Validating {split}",
            unit="sequence",
            dynamic_ncols=True,
            position=0,
            leave=False,
            disable=not SHOW_PROGRESS,
        ):
            validate_sequence(
                xml_path,
                statistics,
                errors,
            )

    if "train" in split_sequences and "test" in split_sequences:
        overlap = (
            split_sequences["train"]
            & split_sequences["test"]
        )

        if overlap:
            errors.append(
                f"Train/test overlap: {sorted(overlap)}"
            )

    coverage = analyze_frame_coverage(split_sequences)

    warnings = []

    if coverage["unannotated_images"] > 0:
        message = (
            f"{coverage['unannotated_images']} images "
            "have no corresponding XML frame."
        )

        if ALLOW_UNANNOTATED_IMAGES:
            warnings.append(message)
        else:
            errors.append(message)

    report = {
        "status": "PASS" if not errors else "FAIL",
        "statistics": {
            **statistics,
            "class_distribution": dict(
                statistics["class_distribution"]
            ),
            "unknown_classes": dict(
                statistics["unknown_classes"]
            ),
        },
        "frame_coverage": coverage,
        "warnings": warnings,
        "errors": errors,
    }

    return report


def save_report(report: dict) -> None:
    DATASET_VALIDATION_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with DATASET_VALIDATION_REPORT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
           ensure_ascii=False,
        )


def main() -> None:
    print("=" * 60)
    print("UA-DETRAC Dataset Validation")
    print("=" * 60)

    report = validate_dataset()
    save_report(report)

    statistics = report["statistics"]
    
    coverage = report["frame_coverage"]

    print(f"Sequences: {statistics['sequences']:,}")
    print(f"Frames: {statistics['frames']:,}")
    print(f"Annotations: {statistics['annotations']:,}")
    print(
        f"Unique vehicles: "
        f"{statistics['unique_vehicles']:,}"
    )
    print(
        f"Invalid boxes: "
        f"{statistics['invalid_boxes']:,}"
    )

    print("\nVehicle classes:")

    for vehicle_type, count in (
        statistics["class_distribution"].items()
    ):
        print(f"  {vehicle_type}: {count:,}")

    print("\n" + "=" * 60)
    print(f"Validation status: {report['status']}")
    print(f"Errors: {len(report['errors']):,}")
    print("=" * 60)

    print(
        f"Report saved to: "
        f"{DATASET_VALIDATION_REPORT}"
    )
    
    print("\nFrame coverage:")
    print(f"  Total images: {coverage['total_images']:,}")
    print(f"  Annotated frames: {coverage['annotated_frames']:,}")
    print(
        f"  Unannotated images: "
        f"{coverage['unannotated_images']:,}"
    )

    print(f"\nWarnings: {len(report['warnings'])}")

    for warning in report["warnings"]:
        print(f"  - {warning}")

    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()