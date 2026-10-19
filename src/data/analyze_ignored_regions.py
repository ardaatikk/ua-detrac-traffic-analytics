from collections import Counter
import json
import xml.etree.ElementTree as ET

from src.config.paths import (
    RAW_DATA_DIR,
    IGNORED_REGION_REPORT,
)

from src.config.dataset import (
    ANNOTATION_FILES,
    IGNORE_ANALYSIS_SPLIT,
    IGNORE_OVERLAP_THRESHOLD,
    BOUNDING_BOX_ATTRIBUTES,
    IGNORE_OVERLAP_BINS,
)


def parse_box(element):
    left, top, width, height = (
        float(element.attrib[name])
        for name in BOUNDING_BOX_ATTRIBUTES
    )

    return (
        left,
        top,
        left + width,
        top + height,
    )


def calculate_overlap_ratio(vehicle_box, ignored_box):
    vx1, vy1, vx2, vy2 = vehicle_box
    ix1, iy1, ix2, iy2 = ignored_box

    intersection_width = max(
        0.0,
        min(vx2, ix2) - max(vx1, ix1),
    )

    intersection_height = max(
        0.0,
        min(vy2, iy2) - max(vy1, iy1),
    )

    intersection_area = (
        intersection_width * intersection_height
    )

    vehicle_area = (
        max(0.0, vx2 - vx1)
        * max(0.0, vy2 - vy1)
    )

    if vehicle_area == 0:
        return 0.0

    return intersection_area / vehicle_area


def analyze_sequence(xml_path):
    root = ET.parse(xml_path).getroot()

    ignored_region = root.find("ignored_region")

    ignored_boxes = []

    if ignored_region is not None:
        ignored_boxes = [
            parse_box(box)
            for box in ignored_region.findall("box")
        ]

    total_vehicles = 0
    overlapping_vehicles = 0
    max_overlap = 0.0
    overlap_distribution = Counter()

    for target in root.findall(
        "./frame/target_list/target"
    ):
        box = target.find("box")

        if box is None:
            continue

        vehicle_box = parse_box(box)
        total_vehicles += 1

        overlap = max(
            (
                calculate_overlap_ratio(
                    vehicle_box,
                    ignored_box,
                )
                for ignored_box in ignored_boxes
            ),
            default=0.0,
        )

        max_overlap = max(max_overlap, overlap)
        for threshold in IGNORE_OVERLAP_BINS:
            if overlap >= threshold:
                overlap_distribution[str(threshold)] += 1

        if overlap >= IGNORE_OVERLAP_THRESHOLD:
            overlapping_vehicles += 1

    return {
        "sequence": root.attrib["name"],
        "ignored_regions": len(ignored_boxes),
        "vehicles": total_vehicles,
        "overlapping_vehicles": overlapping_vehicles,
        "max_overlap_ratio": max_overlap,
        "overlap_distribution": {
            str(threshold): overlap_distribution[str(threshold)]
            for threshold in IGNORE_OVERLAP_BINS
        },
    }


def main():
    annotation_dir = (
        RAW_DATA_DIR
        / ANNOTATION_FILES[IGNORE_ANALYSIS_SPLIT]["directory"]
    )

    results = [
        analyze_sequence(xml_path)
        for xml_path in sorted(annotation_dir.glob("*.xml"))
    ]

    report = {
        "threshold": IGNORE_OVERLAP_THRESHOLD,
        "sequences": results,
    }

    IGNORED_REGION_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with IGNORED_REGION_REPORT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    total_vehicles = sum(
        item["vehicles"]
        for item in results
    )

    overlapping = sum(
        item["overlapping_vehicles"]
        for item in results
    )

    print("=" * 60)
    print("UA-DETRAC Ignored Region Analysis")
    print("=" * 60)

    print(f"Sequences analyzed: {len(results)}")
    print(f"Vehicle annotations: {total_vehicles:,}")
    print(f"Overlapping annotations: {overlapping:,}")

    if total_vehicles:
        percentage = overlapping / total_vehicles * 100
        print(f"Overlap percentage: {percentage:.4f}%")

    print(
        "Maximum overlap ratio:",
        f"{max((r['max_overlap_ratio'] for r in results), default=0):.4f}",
    )

    print("\nOverlap distribution:")
    print("-" * 40)

    for threshold in IGNORE_OVERLAP_BINS:
        count = sum(
            item["overlap_distribution"][str(threshold)]
            for item in results
        )

        percentage = (
            count / total_vehicles * 100
            if total_vehicles
            else 0.0
        )

        print(
            f">= {threshold:.0%}: "
            f"{count:,} annotations "
            f"({percentage:.2f}%)"
        )

    print(f"Report saved to: {IGNORED_REGION_REPORT}")


if __name__ == "__main__":
    main()