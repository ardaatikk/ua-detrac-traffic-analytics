import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from src.config.paths import (
    DATASET_SPLIT_FILE,
    SEQUENCE_STATISTICS_REPORT,
    YOLO_DATASET_DIR,
    YOLO_DATASET_YAML,
    YOLO_CONVERSION_REPORT,
    YOLO_VALIDATION_REPORT,
)


def run_module(
    module_name: str,
    no_progress: bool = False,
    extra_env: dict | None = None,
) -> float:
    environment = os.environ.copy()

    if no_progress:
        environment["UA_DETRAC_SHOW_PROGRESS"] = "0"

    if extra_env:
        environment.update(extra_env)

    command = [
        sys.executable,
        "-m",
        module_name,
    ]

    start_time = time.perf_counter()
    subprocess.run(
        command,
        check=True,
        env=environment,
    )

    return time.perf_counter() - start_time


def prepare_split(no_progress: bool = False) -> str:
    """Reuse the existing split or generate it if missing."""

    if DATASET_SPLIT_FILE.is_file():
        print(
            "Existing dataset split found. Reusing it.",
            flush=True,
        )
        return "REUSED"

    if not SEQUENCE_STATISTICS_REPORT.is_file():
        run_module(
            "src.data.analyze_sequences",
            no_progress=no_progress,
        )

    run_module(
        "src.data.split_dataset",
        no_progress=no_progress,
    )

    if not DATASET_SPLIT_FILE.is_file():
        raise RuntimeError(
            "Dataset split generation failed."
        )

    return "CREATED"


def yolo_outputs_ready() -> bool:
    """Check whether previous YOLO outputs passed validation."""

    required_files = (
        YOLO_DATASET_YAML,
        YOLO_CONVERSION_REPORT,
        YOLO_VALIDATION_REPORT,
    )

    if not all(path.is_file() for path in required_files):
        return False

    try:
        with YOLO_VALIDATION_REPORT.open(
            encoding="utf-8"
        ) as file:
            report = json.load(file)

        return report.get("status") == "PASS"

    except (OSError, ValueError):
        return False


def prepare_yolo(
    force: bool = False,
    no_progress: bool = False,
) -> str:
    if not force and yolo_outputs_ready():
        print(
            "Previously validated YOLO dataset found. "
            "Skipping conversion.",
            flush=True,
        )
        return "REUSED"

    if not force:
        run_module(
            "src.data.convert_to_yolo",
            no_progress=no_progress,
        )
        return "CREATED"

    transaction_id = uuid.uuid4().hex

    staging_dir = YOLO_DATASET_DIR.with_name(
        f"{YOLO_DATASET_DIR.name}.staging-{transaction_id}"
    )

    staging_conversion_report = YOLO_CONVERSION_REPORT.with_name(
        f"{YOLO_CONVERSION_REPORT.stem}.staging-{transaction_id}.json"
    )

    staging_validation_report = YOLO_VALIDATION_REPORT.with_name(
        f"{YOLO_VALIDATION_REPORT.stem}.staging-{transaction_id}.json"
    )

    staging_env = {
        "UA_DETRAC_YOLO_DIR": str(staging_dir),
        "UA_DETRAC_YOLO_FINAL_DIR": str(YOLO_DATASET_DIR),
        "UA_DETRAC_CONVERSION_REPORT": str(staging_conversion_report),
        "UA_DETRAC_VALIDATION_REPORT": str(staging_validation_report),
    }

    # Each item: (new version, published version).
    publications = [
        (staging_dir, YOLO_DATASET_DIR),
        (staging_conversion_report, YOLO_CONVERSION_REPORT),
        (staging_validation_report, YOLO_VALIDATION_REPORT),
    ]

    backups = []
    published = []

    print(
        "Building YOLO dataset in staging directory.",
        flush=True,
    )

    try:
        # Phase 1: Build and validate without touching existing data.
        run_module(
            "src.data.convert_to_yolo",
            no_progress=no_progress,
            extra_env=staging_env,
        )

        run_module(
            "src.data.validate_yolo",
            no_progress=no_progress,
            extra_env=staging_env,
        )

        with staging_validation_report.open(
            encoding="utf-8"
        ) as file:
            validation = json.load(file)

        if validation.get("status") != "PASS":
            raise RuntimeError(
                "Staging dataset validation failed."
            )

        for staged_path, _ in publications:
            if not staged_path.exists():
                raise RuntimeError(
                    f"Missing staging output: {staged_path}"
                )

        # Phase 2: Back up existing published outputs.
        for _, published_path in publications:
            backup_path = published_path.with_name(
                f"{published_path.name}.backup-{transaction_id}"
            )

            if published_path.exists():
                published_path.rename(backup_path)
                backups.append(
                    (backup_path, published_path)
                )

        # Phase 3: Publish validated outputs.
        for staged_path, published_path in publications:
            staged_path.rename(published_path)
            published.append(published_path)

    except Exception:
        print(
            "YOLO rebuild failed. Attempting rollback.",
            flush=True,
        )

        rollback_errors = []

        # Remove newly published outputs.
        for published_path in reversed(published):
            try:
                if published_path.is_dir():
                    shutil.rmtree(published_path)
                elif published_path.exists():
                    published_path.unlink()
            except OSError as exc:
                rollback_errors.append(str(exc))

        # Restore previous outputs.
        for backup_path, original_path in reversed(backups):
            try:
                backup_path.rename(original_path)
            except OSError as exc:
                rollback_errors.append(str(exc))

        if rollback_errors:
            raise RuntimeError(
                "Rollback incomplete: "
                + "; ".join(rollback_errors)
            )

        raise

    finally:
        # Staging outputs are disposable.
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

        staging_conversion_report.unlink(missing_ok=True)
        staging_validation_report.unlink(missing_ok=True)

    # Phase 4: Cleanup after successful publication.
    # Backup deletion must not trigger rollback.
    for backup_path, _ in backups:
        try:
            if backup_path.is_dir():
                shutil.rmtree(backup_path)
            elif backup_path.exists():
                backup_path.unlink()
        except OSError as exc:
            print(
                f"Warning: Could not remove backup "
                f"{backup_path}: {exc}",
                flush=True,
            )

    print(
        "Staging dataset promoted successfully.",
        flush=True,
    )

    return "CREATED"


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the UA-DETRAC dataset."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rebuild processed YOLO files without "
            "deleting raw data or dataset splits."
        ),
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars in supported modules.",
    )

    args = parser.parse_args()

    start_time = time.perf_counter()

    print("=" * 60)
    print("UA-DETRAC | Dataset Preparation Pipeline")
    print("=" * 60)

    steps = [
        "Downloading / checking raw dataset",
        "Validating raw dataset",
        "Preparing dataset split",
        "Converting XML to YOLO",
        "Validating YOLO dataset",
    ]

    results = []

    try:
        for index, description in enumerate(steps, start=1):
            print(
                f"\n[{index}/{len(steps)}] {description}",
                flush=True,
            )
            print("-" * 60, flush=True)

            step_start = time.perf_counter()

            if index == 1:
                run_module(
                    "src.data.download",
                    no_progress=args.no_progress,
                )
                status = "PASSED"

            elif index == 2:
                run_module(
                    "src.data.validate",
                    no_progress=args.no_progress,
                )
                status = "PASSED"

            elif index == 3:
                status = prepare_split(
                    no_progress=args.no_progress,
                )

            elif index == 4:
                status = prepare_yolo(
                    force=args.force,
                    no_progress=args.no_progress,
                )

            elif index == 5:
                run_module(
                    "src.data.validate_yolo",
                    no_progress=args.no_progress,
                )
                status = "PASSED"

            elapsed = time.perf_counter() - step_start

            results.append(
                {
                    "step": description,
                    "status": status,
                    "seconds": elapsed,
                }
            )

            print(
                f"Status: {status} ({elapsed:.2f}s)",
                flush=True,
            )

    except (subprocess.CalledProcessError, Exception) as exc:
        print(
            f"\nPipeline failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    total_time = time.perf_counter() - start_time

    print("\n" + "=" * 60)
    print("Dataset Preparation Summary")
    print("=" * 60)

    for result in results:
        print(
            f"{result['step']}: "
            f"{result['status']} "
            f"({result['seconds']:.2f}s)"
        )

    print("-" * 60)
    print(f"Total time: {total_time:.2f}s")
    print("Status: SUCCESS")
    print("=" * 60)


if __name__ == "__main__":
    main()