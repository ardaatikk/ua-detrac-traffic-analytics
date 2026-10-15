from pathlib import Path
from zipfile import ZipFile, BadZipFile

import gdown

from src.config.paths import (
    RAW_DATA_DIR,
    IMAGES_DIR,
)

from src.config.dataset import (
    IMAGES_FILE_ID,
    IMAGES_ARCHIVE_NAME,
    ANNOTATION_FILES,
    EXPECTED_SEQUENCES,
    EXPECTED_FRAMES,
    IMAGE_EXTENSIONS,
)


def download_file(file_id: str, destination: Path) -> None:
    url = f"https://drive.google.com/uc?id={file_id}"

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Downloading {destination.name}...")

    result = gdown.download(
        url=url,
        output=str(destination),
        quiet=False,
        resume=True,
    )

    if result is None or not destination.exists():
        raise RuntimeError(
            f"Download failed: {destination.name}"
        )


def validate_archive(archive_path: Path) -> None:
    try:
        with ZipFile(archive_path) as archive:
            corrupted_file = archive.testzip()

            if corrupted_file is not None:
                raise RuntimeError(
                    f"Corrupted ZIP member: {corrupted_file}"
                )

    except BadZipFile as exc:
        raise RuntimeError(
            f"Invalid ZIP archive: {archive_path}"
        ) from exc


def extract_archive(archive_path: Path) -> None:
    print(f"Extracting {archive_path.name}...")

    with ZipFile(archive_path) as archive:
        archive.extractall(RAW_DATA_DIR)


def count_images() -> tuple[int, int]:
    if not IMAGES_DIR.exists():
        return 0, 0

    sequence_dirs = [
        path
        for path in IMAGES_DIR.iterdir()
        if path.is_dir()
    ]

    frame_count = sum(
        1
        for path in IMAGES_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    return len(sequence_dirs), frame_count


def prepare_images() -> None:
    sequence_count, frame_count = count_images()

    if (
        sequence_count == EXPECTED_SEQUENCES
        and frame_count == EXPECTED_FRAMES
    ):
        print(
            f"Images ready: {sequence_count} sequences, "
            f"{frame_count} frames."
        )
        return

    archive_path = RAW_DATA_DIR / IMAGES_ARCHIVE_NAME

    if not archive_path.exists():
        download_file(
            IMAGES_FILE_ID,
            archive_path,
        )

    validate_archive(archive_path)
    extract_archive(archive_path)

    sequence_count, frame_count = count_images()

    if (
        sequence_count != EXPECTED_SEQUENCES
        or frame_count != EXPECTED_FRAMES
    ):
        raise RuntimeError(
            "Image dataset validation failed. "
            f"Found {sequence_count} sequences and "
            f"{frame_count} frames."
        )

    print("Images ready.")


def prepare_annotations() -> None:
    for split, config in ANNOTATION_FILES.items():
        annotation_dir = (
            RAW_DATA_DIR / config["directory"]
        )

        archive_path = (
            RAW_DATA_DIR / config["archive"]
        )

        expected_count = config["expected_count"]

        if annotation_dir.exists():
            xml_count = len(
                list(annotation_dir.glob("*.xml"))
            )

            if xml_count == expected_count:
                print(
                    f"{split.capitalize()} annotations ready: "
                    f"{xml_count} XML files."
                )
                continue

        if not archive_path.exists():
            download_file(
                config["file_id"],
                archive_path,
            )

        validate_archive(archive_path)
        extract_archive(archive_path)

        xml_count = len(
            list(annotation_dir.glob("*.xml"))
        )

        if xml_count != expected_count:
            raise RuntimeError(
                f"{split} annotation validation failed. "
                f"Expected {expected_count}, found {xml_count}."
            )

        print(
            f"{split.capitalize()} annotations ready: "
            f"{xml_count} XML files."
        )


def main() -> None:
    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("UA-DETRAC Dataset Preparation")
    print("=" * 60)

    prepare_images()
    prepare_annotations()

    print("=" * 60)
    print("Dataset preparation completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()