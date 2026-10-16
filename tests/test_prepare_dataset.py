import json
import subprocess
from pathlib import Path
import pytest

from src.pipeline import prepare_dataset as pipeline


def test_existing_split_is_reused(tmp_path, monkeypatch):
    split_file = tmp_path / "dataset_split.json"
    split_file.write_text('{"train": [], "val": []}')

    monkeypatch.setattr(
        pipeline, "DATASET_SPLIT_FILE", split_file
    )

    def unexpected_call(*args, **kwargs):
        pytest.fail("Split generation should not run")

    monkeypatch.setattr(
        pipeline, "run_module", unexpected_call
    )

    assert pipeline.prepare_split() == "REUSED"


def test_missing_split_is_created(tmp_path, monkeypatch):
    split_file = tmp_path / "dataset_split.json"
    stats_file = tmp_path / "sequence_statistics.json"

    monkeypatch.setattr(
        pipeline, "DATASET_SPLIT_FILE", split_file
    )
    monkeypatch.setattr(
        pipeline, "SEQUENCE_STATISTICS_REPORT", stats_file
    )

    calls = []

    def fake_run_module(module_name, no_progress=False):
        calls.append(module_name)

        if module_name == "src.data.analyze_sequences":
            stats_file.write_text("{}")

        if module_name == "src.data.split_dataset":
            split_file.write_text("{}")

    monkeypatch.setattr(
        pipeline, "run_module", fake_run_module
    )

    assert pipeline.prepare_split() == "CREATED"

    assert calls == [
        "src.data.analyze_sequences",
        "src.data.split_dataset",
    ]


def test_valid_yolo_outputs_are_reused(tmp_path, monkeypatch):
    yaml_file = tmp_path / "dataset.yaml"
    conversion_report = tmp_path / "conversion.json"
    validation_report = tmp_path / "validation.json"

    yaml_file.write_text("names: []")
    conversion_report.write_text("{}")
    validation_report.write_text(
        json.dumps({"status": "PASS"})
    )

    monkeypatch.setattr(
        pipeline, "YOLO_DATASET_YAML", yaml_file
    )
    monkeypatch.setattr(
        pipeline, "YOLO_CONVERSION_REPORT", conversion_report
    )
    monkeypatch.setattr(
        pipeline, "YOLO_VALIDATION_REPORT", validation_report
    )

    assert pipeline.yolo_outputs_ready() is True


def test_failed_yolo_validation_is_not_reused(
    tmp_path, monkeypatch
):
    yaml_file = tmp_path / "dataset.yaml"
    conversion_report = tmp_path / "conversion.json"
    validation_report = tmp_path / "validation.json"

    yaml_file.write_text("names: []")
    conversion_report.write_text("{}")
    validation_report.write_text(
        json.dumps({"status": "FAIL"})
    )

    monkeypatch.setattr(
        pipeline, "YOLO_DATASET_YAML", yaml_file
    )
    monkeypatch.setattr(
        pipeline, "YOLO_CONVERSION_REPORT", conversion_report
    )
    monkeypatch.setattr(
        pipeline, "YOLO_VALIDATION_REPORT", validation_report
    )

    assert pipeline.yolo_outputs_ready() is False


def test_no_progress_is_forwarded(monkeypatch):
    captured = {}

    def fake_subprocess_run(command, check, env):
        captured["command"] = command
        captured["env"] = env

    monkeypatch.setattr(
        subprocess, "run", fake_subprocess_run
    )

    pipeline.run_module(
        "src.data.validate",
        no_progress=True,
    )

    assert (
        captured["env"]["UA_DETRAC_SHOW_PROGRESS"]
        == "0"
    )


def test_force_preserves_raw_data_and_split(
    tmp_path, monkeypatch
):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    raw_file = raw_dir / "image.jpg"
    raw_file.write_bytes(b"original image")

    split_file = tmp_path / "dataset_split.json"
    split_file.write_text('{"train": [], "val": []}')

    yolo_dir = tmp_path / "yolo"
    yolo_dir.mkdir()

    original_file = yolo_dir / "original.txt"
    original_file.write_text("original dataset")

    conversion_report = tmp_path / "conversion.json"
    validation_report = tmp_path / "validation.json"

    conversion_report.write_text('{"version": "old"}')
    validation_report.write_text('{"status": "PASS"}')

    monkeypatch.setattr(
        pipeline, "YOLO_DATASET_DIR", yolo_dir
    )
    monkeypatch.setattr(
        pipeline, "YOLO_CONVERSION_REPORT", conversion_report
    )
    monkeypatch.setattr(
        pipeline, "YOLO_VALIDATION_REPORT", validation_report
    )

    calls = []

    def fake_run_module(
        module_name,
        no_progress=False,
        extra_env=None,
    ):
        calls.append(module_name)

        staging_dir = Path(
            extra_env["UA_DETRAC_YOLO_DIR"]
        )

        if module_name == "src.data.convert_to_yolo":
            staging_dir.mkdir(parents=True)

            (staging_dir / "new.txt").write_text(
                "new dataset"
            )

            Path(
                extra_env["UA_DETRAC_CONVERSION_REPORT"]
            ).write_text('{"version": "new"}')

        elif module_name == "src.data.validate_yolo":
            Path(
                extra_env["UA_DETRAC_VALIDATION_REPORT"]
            ).write_text('{"status": "PASS"}')

    monkeypatch.setattr(
        pipeline, "run_module", fake_run_module
    )

    result = pipeline.prepare_yolo(force=True)

    assert result == "CREATED"

    # Raw data and dataset split must remain untouched.
    assert raw_file.read_bytes() == b"original image"
    assert split_file.read_text() == '{"train": [], "val": []}'

    # New dataset must replace the old one.
    assert (yolo_dir / "new.txt").read_text() == "new dataset"
    assert not (yolo_dir / "original.txt").exists()

    # Reports must reflect the new dataset.
    assert json.loads(conversion_report.read_text()) == {
        "version": "new"
    }

    assert json.loads(validation_report.read_text()) == {
        "status": "PASS"
    }

    assert calls == [
        "src.data.convert_to_yolo",
        "src.data.validate_yolo",
    ]
    
def test_force_preserves_old_dataset_on_conversion_failure(
    tmp_path, monkeypatch
):
    yolo_dir = tmp_path / "yolo"
    yolo_dir.mkdir()

    original_file = yolo_dir / "original.txt"
    original_file.write_text("original dataset")

    conversion_report = tmp_path / "conversion.json"
    validation_report = tmp_path / "validation.json"

    conversion_report.write_text('{"version": "old"}')
    validation_report.write_text('{"status": "PASS"}')

    monkeypatch.setattr(
        pipeline, "YOLO_DATASET_DIR", yolo_dir
    )
    monkeypatch.setattr(
        pipeline, "YOLO_CONVERSION_REPORT", conversion_report
    )
    monkeypatch.setattr(
        pipeline, "YOLO_VALIDATION_REPORT", validation_report
    )

    def fake_run_module(
        module_name,
        no_progress=False,
        extra_env=None,
    ):
        staging_dir = Path(
            extra_env["UA_DETRAC_YOLO_DIR"]
        )

        staging_dir.mkdir(parents=True)

        (staging_dir / "partial.txt").write_text(
            "incomplete dataset"
        )

        raise RuntimeError("Simulated conversion failure")

    monkeypatch.setattr(
        pipeline, "run_module", fake_run_module
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated conversion failure",
    ):
        pipeline.prepare_yolo(force=True)

    # Existing dataset must survive.
    assert original_file.read_text() == "original dataset"

    # Existing reports must survive.
    assert json.loads(conversion_report.read_text()) == {
        "version": "old"
    }

    assert json.loads(validation_report.read_text()) == {
        "status": "PASS"
    }

    # Temporary staging directories must be removed.
    assert not list(tmp_path.glob("yolo.staging-*"))
    
def test_force_preserves_old_dataset_on_validation_failure(
    tmp_path, monkeypatch
):
    yolo_dir = tmp_path / "yolo"
    yolo_dir.mkdir()

    original_file = yolo_dir / "original.txt"
    original_file.write_text("original dataset")

    conversion_report = tmp_path / "conversion.json"
    validation_report = tmp_path / "validation.json"

    conversion_report.write_text('{"version": "old"}')
    validation_report.write_text('{"status": "PASS"}')

    monkeypatch.setattr(
        pipeline, "YOLO_DATASET_DIR", yolo_dir
    )
    monkeypatch.setattr(
        pipeline, "YOLO_CONVERSION_REPORT", conversion_report
    )
    monkeypatch.setattr(
        pipeline, "YOLO_VALIDATION_REPORT", validation_report
    )

    def fake_run_module(
        module_name,
        no_progress=False,
        extra_env=None,
    ):
        if module_name == "src.data.convert_to_yolo":
            staging_dir = Path(
                extra_env["UA_DETRAC_YOLO_DIR"]
            )
            staging_dir.mkdir()

            (staging_dir / "new.txt").write_text(
                "new dataset"
            )

            Path(
                extra_env["UA_DETRAC_CONVERSION_REPORT"]
            ).write_text('{"version": "new"}')

        elif module_name == "src.data.validate_yolo":
            Path(
                extra_env["UA_DETRAC_VALIDATION_REPORT"]
            ).write_text('{"status": "FAIL"}')

    monkeypatch.setattr(
        pipeline, "run_module", fake_run_module
    )

    with pytest.raises(
        RuntimeError,
        match="Staging dataset validation failed",
    ):
        pipeline.prepare_yolo(force=True)

    assert original_file.read_text() == "original dataset"

    assert json.loads(conversion_report.read_text()) == {
        "version": "old"
    }

    assert json.loads(validation_report.read_text()) == {
        "status": "PASS"
    }

    assert not list(tmp_path.glob("yolo.staging-*"))
    
def test_force_rolls_back_on_report_publication_failure(
    tmp_path, monkeypatch
):
    yolo_dir = tmp_path / "yolo"
    yolo_dir.mkdir()

    (yolo_dir / "original.txt").write_text("original dataset")

    conversion_report = tmp_path / "conversion.json"
    validation_report = tmp_path / "validation.json"

    conversion_report.write_text('{"version": "old"}')
    validation_report.write_text('{"status": "PASS"}')

    monkeypatch.setattr(
        pipeline, "YOLO_DATASET_DIR", yolo_dir
    )
    monkeypatch.setattr(
        pipeline, "YOLO_CONVERSION_REPORT", conversion_report
    )
    monkeypatch.setattr(
        pipeline, "YOLO_VALIDATION_REPORT", validation_report
    )

    def fake_run_module(
        module_name,
        no_progress=False,
        extra_env=None,
    ):
        if module_name == "src.data.convert_to_yolo":
            staging_dir = Path(
                extra_env["UA_DETRAC_YOLO_DIR"]
            )
            staging_dir.mkdir()

            (staging_dir / "new.txt").write_text(
                "new dataset"
            )

            Path(
                extra_env["UA_DETRAC_CONVERSION_REPORT"]
            ).write_text('{"version": "new"}')

        elif module_name == "src.data.validate_yolo":
            Path(
                extra_env["UA_DETRAC_VALIDATION_REPORT"]
            ).write_text('{"status": "PASS"}')

    monkeypatch.setattr(
        pipeline, "run_module", fake_run_module
    )

    original_rename = Path.rename

    def failing_rename(self, target):
        target = Path(target)

        # Simulate failure when publishing the new validation report.
        if (
            self.name.startswith("validation.staging-")
            and target == validation_report
        ):
            raise OSError("Simulated publication failure")

        return original_rename(self, target)

    monkeypatch.setattr(
        Path, "rename", failing_rename
    )

    with pytest.raises(
        OSError,
        match="Simulated publication failure",
    ):
        pipeline.prepare_yolo(force=True)

    # All previous outputs must be restored.
    assert (yolo_dir / "original.txt").read_text() == (
        "original dataset"
    )

    assert not (yolo_dir / "new.txt").exists()

    assert json.loads(conversion_report.read_text()) == {
        "version": "old"
    }

    assert json.loads(validation_report.read_text()) == {
        "status": "PASS"
    }

    assert not list(tmp_path.glob("*.backup-*"))
    assert not list(tmp_path.glob("yolo.staging-*"))