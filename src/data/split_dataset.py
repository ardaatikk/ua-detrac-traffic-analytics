import json

import numpy as np

from src.config.paths import (
    SEQUENCE_STATISTICS_REPORT,
    DATASET_SPLIT_FILE,
)

from src.config.dataset import VEHICLE_CLASSES

from src.config.settings import (
    RANDOM_SEED,
    VALIDATION_RATIO,
    SPLIT_SEARCH_ITERATIONS,
    SPLIT_CLASS_BALANCE_WEIGHT,
    SPLIT_FRAME_BALANCE_WEIGHT,
    SPLIT_MIN_CLASS_ANNOTATIONS,
)


def load_sequence_statistics():
    with SEQUENCE_STATISTICS_REPORT.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def calculate_distribution(sequences):
    counts = np.array(
        [
            sum(
                sequence["classes"][class_name]
                for sequence in sequences
            )
            for class_name in VEHICLE_CLASSES
        ],
        dtype=np.float64,
    )

    total = counts.sum()

    if total == 0:
        return np.zeros_like(counts)

    return counts / total


def calculate_score(train_sequences, val_sequences):
    train_distribution = calculate_distribution(
        train_sequences
    )

    val_distribution = calculate_distribution(
        val_sequences
    )

    class_difference = np.abs(
        train_distribution - val_distribution
    ).sum()

    total_frames = sum(
        sequence["frames"]
        for sequence in train_sequences + val_sequences
    )

    val_frames = sum(
        sequence["frames"]
        for sequence in val_sequences
    )

    frame_ratio = val_frames / total_frames

    frame_difference = abs(
        frame_ratio - VALIDATION_RATIO
    )

    score = (
        SPLIT_CLASS_BALANCE_WEIGHT * class_difference
        + SPLIT_FRAME_BALANCE_WEIGHT * frame_difference
    )

    return float(score)


def has_all_classes(sequences):
    return all(
        sum(
            sequence["classes"][class_name]
            for sequence in sequences
        ) >= SPLIT_MIN_CLASS_ANNOTATIONS
        for class_name in VEHICLE_CLASSES
    )


def find_best_split(statistics):
    rng = np.random.default_rng(RANDOM_SEED)

    sequence_count = len(statistics)

    val_count = round(
        sequence_count * VALIDATION_RATIO
    )

    if not 0 < val_count < sequence_count:
        raise ValueError(
            "Validation ratio produces an invalid split."
        )

    best_score = float("inf")
    best_train = None
    best_val = None

    for _ in range(SPLIT_SEARCH_ITERATIONS):
        indices = rng.permutation(sequence_count)

        val_indices = set(indices[:val_count])

        train_sequences = [
            statistics[index]
            for index in range(sequence_count)
            if index not in val_indices
        ]

        val_sequences = [
            statistics[index]
            for index in range(sequence_count)
            if index in val_indices
        ]

        if not has_all_classes(train_sequences):
            continue

        if not has_all_classes(val_sequences):
            continue

        score = calculate_score(
            train_sequences,
            val_sequences,
        )

        if score < best_score:
            best_score = score
            best_train = train_sequences
            best_val = val_sequences

    if best_train is None or best_val is None:
        raise RuntimeError(
            "No valid split found. "
            "Try increasing SPLIT_SEARCH_ITERATIONS."
        )

    return best_train, best_val, best_score


def summarize_split(sequences):
    return {
        "sequence_count": len(sequences),
        "frame_count": sum(
            sequence["frames"]
            for sequence in sequences
        ),
        "annotation_count": sum(
            sequence["annotations"]
            for sequence in sequences
        ),
        "class_counts": {
            class_name: sum(
                sequence["classes"][class_name]
                for sequence in sequences
            )
            for class_name in VEHICLE_CLASSES
        },
    }


def save_split(train_sequences, val_sequences, score):
    train_names = sorted(
        sequence["sequence"]
        for sequence in train_sequences
    )

    val_names = sorted(
        sequence["sequence"]
        for sequence in val_sequences
    )

    if set(train_names) & set(val_names):
        raise RuntimeError(
            "Train and validation sequences overlap."
        )

    result = {
        "seed": RANDOM_SEED,
        "validation_ratio": VALIDATION_RATIO,
        "score": score,
        "train": train_names,
        "val": val_names,
        "summary": {
            "train": summarize_split(train_sequences),
            "val": summarize_split(val_sequences),
        },
    }

    DATASET_SPLIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with DATASET_SPLIT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=4,
        )

    return result


def main():
    print("=" * 60)
    print("UA-DETRAC Sequence-Level Train/Validation Split")
    print("=" * 60)

    statistics = load_sequence_statistics()

    train_sequences, val_sequences, score = (
        find_best_split(statistics)
    )

    result = save_split(
        train_sequences,
        val_sequences,
        score,
    )

    for split_name in ("train", "val"):
        summary = result["summary"][split_name]

        print(f"\n{split_name.upper()}")
        print("-" * 40)

        print(
            f"Sequences: {summary['sequence_count']}"
        )

        print(
            f"Frames: {summary['frame_count']:,}"
        )

        print(
            f"Annotations: {summary['annotation_count']:,}"
        )

        for class_name, count in (
            summary["class_counts"].items()
        ):
            print(f"  {class_name}: {count:,}")

    print(f"\nBest split score: {score:.6f}")
    print(f"Split saved to: {DATASET_SPLIT_FILE}")


if __name__ == "__main__":
    main()