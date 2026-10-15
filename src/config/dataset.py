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