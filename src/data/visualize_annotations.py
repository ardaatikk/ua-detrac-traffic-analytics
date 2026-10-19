import xml.etree.ElementTree as ET

import cv2

from src.config.paths import (
    RAW_DATA_DIR,
    IMAGES_DIR,
    ANNOTATION_PREVIEW_FILE,
)

from src.config.dataset import (
    ANNOTATION_FILES,
    PREVIEW_SEQUENCE,
    PREVIEW_FRAME,
    PREVIEW_BOX_COLOR,
    PREVIEW_TEXT_COLOR,
    PREVIEW_BOX_THICKNESS,
    PREVIEW_FONT_SCALE,
    FRAME_FILENAME_PREFIX,
    FRAME_FILENAME_DIGITS,
    FRAME_FILENAME_EXTENSION,
    FRAME_NUMBER_ATTRIBUTE,
    VEHICLE_TYPE_ATTRIBUTE,
    TARGET_ID_ATTRIBUTE,
    BOUNDING_BOX_ATTRIBUTES,
    PREVIEW_IGNORED_REGION_COLOR,
    PREVIEW_IGNORED_REGION_THICKNESS,
)

from src.data.convert_to_yolo import convert_bbox_to_yolo


def main():
    annotation_dir = (
        RAW_DATA_DIR
        / ANNOTATION_FILES["train"]["directory"]
    )

    xml_path = annotation_dir / f"{PREVIEW_SEQUENCE}.xml"

    image_name = (
        f"{FRAME_FILENAME_PREFIX}"
        f"{PREVIEW_FRAME:0{FRAME_FILENAME_DIGITS}d}"
        f"{FRAME_FILENAME_EXTENSION}"
    )

    image_path = IMAGES_DIR / PREVIEW_SEQUENCE / image_name

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(image_path)

    image_height, image_width = image.shape[:2]

    root = ET.parse(xml_path).getroot()

    frame = next(
        (
            item
            for item in root.findall("frame")
            if int(item.attrib[FRAME_NUMBER_ATTRIBUTE])
            == PREVIEW_FRAME
        ),
        None,
    )

    if frame is None:
        raise ValueError(
            f"Frame {PREVIEW_FRAME} not found in XML."
        )

    target_list = frame.find("target_list")

    if target_list is None:
        raise ValueError("No target_list found.")

    ignored_region = root.find("ignored_region")

    ignored_count = 0

    if ignored_region is not None:
        for box in ignored_region.findall("box"):
            left, top, width, height = (
                float(box.attrib[name])
                for name in BOUNDING_BOX_ATTRIBUTES
            )

            x1 = max(0, min(round(left), image_width))
            y1 = max(0, min(round(top), image_height))
            x2 = max(0, min(round(left + width), image_width))
            y2 = max(0, min(round(top + height), image_height))

            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                PREVIEW_IGNORED_REGION_COLOR,
                PREVIEW_IGNORED_REGION_THICKNESS,
            )

            ignored_count += 1

    count = 0

    for target in target_list.findall("target"):
        box = target.find("box")
        attribute = target.find("attribute")

        if box is None or attribute is None:
            continue

        left, top, width, height = (
            float(box.attrib[name])
            for name in BOUNDING_BOX_ATTRIBUTES
        )

        try:
            x_center, y_center, norm_width, norm_height = (
                convert_bbox_to_yolo(
                    left,
                    top,
                    width,
                    height,
                    image_width,
                    image_height,
                )
            )
        except ValueError:
            continue

        x1 = int(
            (x_center - norm_width / 2) * image_width
        )
        y1 = int(
            (y_center - norm_height / 2) * image_height
        )
        x2 = int(
            (x_center + norm_width / 2) * image_width
        )
        y2 = int(
            (y_center + norm_height / 2) * image_height
        )

        vehicle_type = attribute.attrib.get(
            VEHICLE_TYPE_ATTRIBUTE,
            "unknown",
        )

        vehicle_id = target.attrib.get(
            TARGET_ID_ATTRIBUTE,
            "?",
        )

        label = f"{vehicle_type} #{vehicle_id}"

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            PREVIEW_BOX_COLOR,
            PREVIEW_BOX_THICKNESS,
        )

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            PREVIEW_FONT_SCALE,
            PREVIEW_TEXT_COLOR,
            1,
        )

        count += 1

    ANNOTATION_PREVIEW_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(ANNOTATION_PREVIEW_FILE),
        image,
    )

    if not success:
        raise RuntimeError("Failed to save preview image.")

    print(f"Sequence: {PREVIEW_SEQUENCE}")
    print(f"Frame: {PREVIEW_FRAME}")
    print(f"Vehicles drawn: {count}")
    print(f"Preview saved: {ANNOTATION_PREVIEW_FILE}")
    print(f"Ignored regions drawn: {ignored_count}")


if __name__ == "__main__":
    main()