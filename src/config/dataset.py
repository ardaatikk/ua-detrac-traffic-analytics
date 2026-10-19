IMAGES_FILE_ID = "1fQC6CEWOeL9pnJ4ZdD7U2gromcnWDRgm"

IMAGES_ARCHIVE_NAME = "DETRAC-Images.zip"

ANNOTATION_FILES = {
    "train": {
        "file_id": "1f22g3puG164gemY_njcPG8beuCcuMjqb",
        "archive": "DETRAC-Train-Annotations-XML.zip",
        "directory": "DETRAC-Train-Annotations-XML",
        "expected_count": 60,
    },
    "test": {
        "file_id": "1zc8KZWUa3Cw0DbcFqTCMBmUZzbfuhGKr",
        "archive": "DETRAC-Test-Annotations-XML.zip",
        "directory": "DETRAC-Test-Annotations-XML",
        "expected_count": 40,
    },
}

EXPECTED_SEQUENCES = 100
EXPECTED_FRAMES = 140_131

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

VEHICLE_CLASSES = {
    "car": 0,
    "bus": 1,
    "van": 2,
    "others": 3,
}

# UA-DETRAC annotation structure
FRAME_NUMBER_ATTRIBUTE = "num"
TARGET_ID_ATTRIBUTE = "id"
VEHICLE_TYPE_ATTRIBUTE = "vehicle_type"

FRAME_FILENAME_PREFIX = "img"
FRAME_FILENAME_DIGITS = 5
FRAME_FILENAME_EXTENSION = ".jpg"

BOUNDING_BOX_ATTRIBUTES = (
    "left",
    "top",
    "width",
    "height",
)

# Dataset validation
EXPECTED_IMAGE_WIDTH = 960
EXPECTED_IMAGE_HEIGHT = 540

# Validation policy
ALLOW_UNANNOTATED_IMAGES = True

# Dataset coverage
REPORT_UNANNOTATED_SEQUENCES = True

# Sequence analysis
ANALYSIS_SPLIT = "train"

# Annotation visualization
PREVIEW_SEQUENCE = "MVI_20011"
PREVIEW_FRAME = 53

PREVIEW_BOX_COLOR = (0, 255, 0)
PREVIEW_TEXT_COLOR = (255, 255, 255)
PREVIEW_BOX_THICKNESS = 2
PREVIEW_FONT_SCALE = 0.5

PREVIEW_IGNORED_REGION_COLOR = (0, 0, 255)
PREVIEW_IGNORED_REGION_THICKNESS = 2

IGNORE_ANALYSIS_SPLIT = "train"
IGNORE_OVERLAP_THRESHOLD = 0.10

IGNORE_OVERLAP_BINS = (
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    1.00,
)

YOLO_SPLITS = ("train", "val", "test")

YOLO_CLASS_NAMES = tuple(
    name
    for name, class_id in sorted(
        VEHICLE_CLASSES.items(),
        key=lambda item: item[1],
    )
)