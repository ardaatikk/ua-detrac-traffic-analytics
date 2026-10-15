from src.config.dataset import (
    EXPECTED_IMAGE_WIDTH,
    EXPECTED_IMAGE_HEIGHT,
)


def convert_bbox_to_yolo(
    left: float,
    top: float,
    width: float,
    height: float,
    image_width: int = EXPECTED_IMAGE_WIDTH,
    image_height: int = EXPECTED_IMAGE_HEIGHT,
) -> tuple[float, float, float, float]:
    """Convert an absolute XYWH bounding box to normalized YOLO format."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    if width <= 0 or height <= 0:
        raise ValueError("Bounding box dimensions must be positive.")

    x_center = (left + width / 2) / image_width
    y_center = (top + height / 2) / image_height

    normalized_width = width / image_width
    normalized_height = height / image_height

    return (
        x_center,
        y_center,
        normalized_width,
        normalized_height,
    )