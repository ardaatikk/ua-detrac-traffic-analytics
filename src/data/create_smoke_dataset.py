"""Build a deterministic, sequence-aware smoke dataset without copying data."""
import argparse
import json
import random
import tempfile
from collections import defaultdict
from pathlib import Path

from src.config.dataset import IMAGE_EXTENSIONS, VEHICLE_CLASSES, YOLO_CLASS_NAMES
from src.config.paths import YOLO_DATASET_DIR, YOLO_SMOKE_DIR
from src.config.settings import RANDOM_SEED, SMOKE_TRAIN_SIZE, SMOKE_VAL_SIZE


def scan_split(source, split):
    records = {}
    for image in sorted((source / 'images' / split).rglob('*')):
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = image.relative_to(source / 'images' / split)
        if len(relative.parts) != 2:
            raise ValueError(f'Expected sequence/frame layout: {image}')
        label = source / 'labels' / split / relative.with_suffix('.txt')
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f'Missing image or label: {image}, {label}')
        classes = set()
        for line in label.read_text().splitlines():
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5 or int(fields[0]) not in VEHICLE_CLASSES.values():
                raise ValueError(f'Invalid YOLO label: {label}')
            classes.add(int(fields[0]))
        records[relative] = classes
    return records


def select_frames(records, size, seed):
    if size <= 0 or size > len(records):
        raise ValueError(f'Requested {size} frames; available {len(records)}')
    required = set(VEHICLE_CLASSES.values())
    selected = []
    covered = set()
    # Cover rare classes first, preferring previously unrepresented sequences.
    while required - covered:
        missing = required - covered
        rare = min(missing, key=lambda c: (sum(c in v for v in records.values()), c))
        candidates = [p for p, classes in records.items() if rare in classes and p not in selected]
        if not candidates:
            raise ValueError(f'Class {rare} absent from split')
        used = {p.parent for p in selected}
        choice = min(candidates, key=lambda p: (p.parent in used, -len(records[p] & missing), str(p)))
        selected.append(choice)
        covered.update(records[choice])
    if len(selected) > size:
        raise ValueError('Requested size cannot cover all classes')
    groups = defaultdict(list)
    for path in sorted(records):
        groups[path.parent].append(path)
    rng = random.Random(seed)
    sequences = sorted(groups)
    rng.shuffle(sequences)
    # Recursively bisect time: first/middle/last and progressively fill gaps.
    queues = {}
    for sequence, paths in groups.items():
        intervals = [(0, len(paths) - 1)]
        order = []
        while intervals:
            left, right = intervals.pop(0)
            if left > right:
                continue
            mid = (left + right) // 2
            order.append(paths[mid])
            intervals.extend([(left, mid - 1), (mid + 1, right)])
        queues[sequence] = iter(order)
    chosen = set(selected)
    while len(selected) < size:
        for sequence in sequences:
            candidate = next((p for p in queues[sequence] if p not in chosen), None)
            if candidate is not None:
                selected.append(candidate)
                chosen.add(candidate)
            if len(selected) == size:
                break
    return sorted(selected)


def create_smoke_dataset(source=None, destination=None, train_size=SMOKE_TRAIN_SIZE,
                         val_size=SMOKE_VAL_SIZE, seed=RANDOM_SEED):
    source = Path(source or YOLO_DATASET_DIR).resolve()
    destination = Path(destination or YOLO_SMOKE_DIR).absolute()
    if source == destination.resolve() or source in destination.resolve().parents or destination.resolve() in source.parents:
        raise ValueError('Source and smoke directories must be separate')
    # Never overwrite existing datasets, symlinks or user files.
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Smoke dataset already exists: {destination}')
    records = {split: scan_split(source, split) for split in ('train', 'val')}
    if {p.parent for p in records['train']} & {p.parent for p in records['val']}:
        raise ValueError('Train and validation sequences overlap')
    selected = {split: select_frames(records[split], size, seed)
                for split, size in (('train', train_size), ('val', val_size))}
    manifest = {'source': str(source), 'seed': seed, 'splits': {}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.smoke-', dir=destination.parent) as temporary:
        staging = Path(temporary) / 'dataset'
        staging.mkdir()
        for split, paths in selected.items():
            counts = {name: 0 for name in YOLO_CLASS_NAMES}
            for relative in paths:
                for kind in ('images', 'labels'):
                    rel = relative if kind == 'images' else relative.with_suffix('.txt')
                    target = staging / kind / split / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # Preserve the YOLO image path instead of resolving through to raw.
                    target.symlink_to(source / kind / split / rel)
                for name, class_id in VEHICLE_CLASSES.items():
                    counts[name] += class_id in records[split][relative]
            manifest['splits'][split] = {'frames': [str(p) for p in paths],
                'sequences': sorted({str(p.parent) for p in paths}),
                'class_image_counts': counts}
        # JSON strings are also valid YAML scalars.
        yaml = f'path: {json.dumps(str(destination))}\ntrain: images/train\nval: images/val\nnames:\n'
        yaml += ''.join(f'  {i}: {json.dumps(name)}\n' for i, name in enumerate(YOLO_CLASS_NAMES))
        (staging / 'dataset.yaml').write_text(yaml)
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        staging.rename(destination)
    return destination / 'dataset.yaml'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-size', type=int, default=SMOKE_TRAIN_SIZE)
    parser.add_argument('--val-size', type=int, default=SMOKE_VAL_SIZE)
    args = parser.parse_args()
    print(create_smoke_dataset(train_size=args.train_size, val_size=args.val_size))


if __name__ == '__main__':
    main()
