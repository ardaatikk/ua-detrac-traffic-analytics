import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"

IMAGES_DIR = RAW_DATA_DIR / "DETRAC-Images"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"

REPORTS_DIR = ARTIFACTS_DIR / "reports"

DATASET_VALIDATION_REPORT = (
    REPORTS_DIR / "dataset_validation.json"
)

SEQUENCE_STATISTICS_REPORT = (
    REPORTS_DIR / "sequence_statistics.json"
)

DATASET_SPLIT_FILE = (
    PROCESSED_DATA_DIR / "dataset_split.json"
)

ANNOTATION_PREVIEW_FILE = (
    FIGURES_DIR / "annotation_preview.png"
)

IGNORED_REGION_REPORT = (
    REPORTS_DIR / "ignored_region_analysis.json"
)

###### YOLO

YOLO_DATASET_DIR = Path(
    os.environ.get(
        "UA_DETRAC_YOLO_DIR",
        str(PROCESSED_DATA_DIR / "yolo"),
    )
).resolve()

YOLO_IMAGES_DIR = YOLO_DATASET_DIR / "images"
YOLO_LABELS_DIR = YOLO_DATASET_DIR / "labels"
YOLO_DATASET_YAML = YOLO_DATASET_DIR / "dataset.yaml"

YOLO_CONVERSION_REPORT = Path(
    os.environ.get(
        "UA_DETRAC_CONVERSION_REPORT",
        str(REPORTS_DIR / "yolo_conversion.json"),
    )
).resolve()

YOLO_VALIDATION_REPORT = Path(
    os.environ.get(
        "UA_DETRAC_VALIDATION_REPORT",
        str(REPORTS_DIR / "yolo_validation.json"),
    )
).resolve()