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