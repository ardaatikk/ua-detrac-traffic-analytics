import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from src.data.create_smoke_dataset import create_smoke_dataset, select_frames


def fixture_data(root):
    for split in ['train', 'val']:
        for seq in range(4):
            for frame in range(12):
                for kind, ext, text in [('images', '.jpg', 'image'), ('labels', '.txt', f'{seq} .5 .5 .2 .2\n')]:
                    p = root / kind / split / f'{split}{seq}' / f'img{frame:05}{ext}'
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(text)


def test_creation(tmp_path):
    source = tmp_path / 'source'
    fixture_data(source)
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    for name in ['a', 'b']:
        yaml = create_smoke_dataset(source, tmp_path / name, 16, 8)
        assert 'test:' not in yaml.read_text()
        report = json.loads(yaml.with_name('manifest.json').read_text())
        for split, count in [('train', 16), ('val', 8)]:
            data = report['splits'][split]
            assert len(data['frames']) == count
            assert len(data['sequences']) == 4
            assert all(data['class_image_counts'].values())
            for relative in data['frames']:
                for kind in ['images', 'labels']:
                    rel = Path(relative) if kind == 'images' else Path(relative).with_suffix('.txt')
                    link = yaml.parent / kind / split / rel
                    assert link.is_symlink() and link.is_file()
    assert (tmp_path / 'a/manifest.json').read_bytes() == (tmp_path / 'b/manifest.json').read_bytes()
    assert all(p.read_bytes() == data for p, data in before.items())
    with pytest.raises(FileExistsError):
        create_smoke_dataset(source, tmp_path / 'a', 16, 8)


def test_bad_selection():
    records = {Path(f'A/img{i:05}.jpg'): {i % 4} for i in range(20)}
    for size in [0, 3, 21]:
        with pytest.raises(ValueError):
            select_frames(records, size, 42)
    with pytest.raises(ValueError, match='absent'):
        select_frames({Path('A/img.jpg'): {0}}, 1, 42)


def test_input_guards(tmp_path):
    source = tmp_path / 'source'
    fixture_data(source)
    with pytest.raises(ValueError, match='separate'):
        create_smoke_dataset(source, source / 'smoke', 8, 8)
    for kind in ['images', 'labels']:
        (source / kind / 'val/val0').rename(source / kind / 'val/train0')
    with pytest.raises(ValueError, match='overlap'):
        create_smoke_dataset(source, tmp_path / 'out', 8, 8)
    next((source / 'labels/train').rglob('*.txt')).unlink()
    with pytest.raises(FileNotFoundError):
        create_smoke_dataset(source, tmp_path / 'out', 8, 8)
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('smoke', [False, True])
def test_training_routes_dataset(tmp_path, monkeypatch, smoke):
    from src.training import train
    from src.config import paths
    full = tmp_path / 'full.yaml'
    full.write_text('')
    small = tmp_path / 'small'
    small.mkdir()
    (small / 'dataset.yaml').write_text('')
    monkeypatch.setattr(train, 'YOLO_DATASET_YAML', full)
    monkeypatch.setattr(paths, 'YOLO_SMOKE_DIR', small)
    monkeypatch.setattr(train, 'TRAINING_DIR', tmp_path / 'runs')
    args = train.parse_args(['--name', 'unit', '--device', 'cpu'] + (['--smoke'] if smoke else []))
    monkeypatch.setattr(train, 'parse_args', lambda: args)
    model = Mock()
    monkeypatch.setattr(train, 'YOLO', Mock(return_value=model))
    train.main()
    assert model.train.call_args.kwargs['data'] == str(small / 'dataset.yaml' if smoke else full)
    if smoke:
        args.fraction = .1
        with pytest.raises(ValueError, match='fraction'):
            train.main()
