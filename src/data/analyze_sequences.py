from collections import Counter
import json
import xml.etree.ElementTree as ET

from src.config.paths import (
    RAW_DATA_DIR,
    SEQUENCE_STATISTICS_REPORT,
)

from src.config.dataset import (
    ANNOTATION_FILES,
    ANALYSIS_SPLIT,
    VEHICLE_CLASSES,
    VEHICLE_TYPE_ATTRIBUTE,
    TARGET_ID_ATTRIBUTE,
)


def analyze_sequence(xml_path):
    root = ET.parse(xml_path).getroot()

    sequence_name = root.attrib["name"]

    class_counts = Counter()
    unique_vehicles = set()
    frame_count = 0

    for frame in root.findall("frame"):
        frame_count += 1

        target_list = frame.find("target_list")

        if target_list is None:
            continue

        for target in target_list.findall("target"):
            target_id = target.attrib.get(
                TARGET_ID_ATTRIBUTE
            )

            if target_id is not None:
                unique_vehicles.add(target_id)

            attribute = target.find("attribute")

            if attribute is None:
                continue

            vehicle_type = attribute.attrib.get(
                VEHICLE_TYPE_ATTRIBUTE
            )

            if vehicle_type in VEHICLE_CLASSES:
                class_counts[vehicle_type] += 1

    return {
        "sequence": sequence_name,
        "frames": frame_count,
        "unique_vehicles": len(unique_vehicles),
        "annotations": sum(class_counts.values()),
        "classes": {
            class_name: class_counts[class_name]
            for class_name in VEHICLE_CLASSES
        },
    }


def analyze_training_sequences():
    config = ANNOTATION_FILES[ANALYSIS_SPLIT]

    annotation_dir = (
        RAW_DATA_DIR / config["directory"]
    )

    results = []

    for xml_path in sorted(annotation_dir.glob("*.xml")):
        results.append(
            analyze_sequence(xml_path)
        )

    return results


def save_statistics(results):
    SEQUENCE_STATISTICS_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SEQUENCE_STATISTICS_REPORT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )


def main():
    print("=" * 60)
    print("UA-DETRAC Sequence Analysis")
    print("=" * 60)

    results = analyze_training_sequences()

    save_statistics(results)

    print(f"Sequences analyzed: {len(results)}")

    print("\nClass distribution across sequences:")
    print("-" * 60)

    for class_name in VEHICLE_CLASSES:
        sequences_with_class = sum(
            result["classes"][class_name] > 0
            for result in results
        )

        total_annotations = sum(
            result["classes"][class_name]
            for result in results
        )

        print(
            f"{class_name:<10} "
            f"Sequences: {sequences_with_class:>2} "
            f"Annotations: {total_annotations:>10,}"
        )

    print("\nSequence statistics saved to:")
    print(SEQUENCE_STATISTICS_REPORT)


if __name__ == "__main__":
    main()