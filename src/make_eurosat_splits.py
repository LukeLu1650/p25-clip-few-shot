"""Create reproducible class-stratified EuroSAT splits and nested few-shot lists.

Run from the project root:
    python src/make_eurosat_splits.py

Outputs are written to data/manifests/. The indices refer to the ordering in
`torchvision.datasets.EuroSAT(...).samples` for the same EuroSAT RGB dataset.
"""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from torchvision.datasets import EuroSAT


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MANIFEST_DIR = DATA_ROOT / "manifests"

SPLIT_SEED = 2026
FEW_SHOT_SEEDS = (1, 2, 3)
SHOT_COUNTS = (1, 2, 4, 8, 16)
TRAIN_RATIO = 0.60
VAL_RATIO = 0.10


def save_json(path: Path, payload: dict) -> None:
    """Write readable deterministic JSON; refuse to silently replace changed data."""
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(
                f"Existing manifest differs: {path}. "
                "Do not overwrite it without coordinating with teammates."
            )
        print(f"Already exists (unchanged): {path.relative_to(PROJECT_ROOT)}")
        return
    path.write_text(encoded, encoding="utf-8")
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    print("Loading EuroSAT RGB dataset...")
    dataset = EuroSAT(root=str(DATA_ROOT), download=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    classes = list(dataset.classes)
    by_class = defaultdict(list)
    fingerprint = hashlib.sha256()
    for index, (image_path, class_id) in enumerate(dataset.samples):
        by_class[class_id].append(index)
        # Detect data ordering changes across machines / dependency versions.
        relative_path = Path(image_path).relative_to(Path(dataset.root)).as_posix()
        fingerprint.update(f"{index}\t{relative_path}\t{class_id}\n".encode("utf-8"))
    dataset_fingerprint = fingerprint.hexdigest()

    rng = random.Random(SPLIT_SEED)
    split_by_class = {}
    class_counts = {}
    for class_id, class_name in enumerate(classes):
        indices = by_class[class_id].copy()
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        split_by_class[class_id] = {
            "train_pool": indices[:n_train],
            "val": indices[n_train : n_train + n_val],
            "test": indices[n_train + n_val :],
        }
        class_counts[class_name] = {
            "total": n,
            "train_pool": n_train,
            "val": n_val,
            "test": n - n_train - n_val,
        }
        if n_train < max(SHOT_COUNTS):
            raise ValueError(f"Insufficient train samples in class {class_name}")

    splits = {
        name: sorted(
            index
            for class_id in range(len(classes))
            for index in split_by_class[class_id][name]
        )
        for name in ("train_pool", "val", "test")
    }
    train_set, val_set, test_set = map(set, (
        splits["train_pool"], splits["val"], splits["test"]
    ))
    assert not (train_set & val_set or train_set & test_set or val_set & test_set)
    assert len(train_set | val_set | test_set) == len(dataset)

    common = {
        "dataset": "EuroSAT RGB (torchvision.datasets.EuroSAT)",
        "dataset_size": len(dataset),
        "classes": classes,
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "split_seed": SPLIT_SEED,
    }
    save_json(
        MANIFEST_DIR / "eurosat_split.json",
        {
            **common,
            "ratios": {
                "train_pool": TRAIN_RATIO,
                "val": VAL_RATIO,
                "test": 1.0 - TRAIN_RATIO - VAL_RATIO,
            },
            "class_counts": class_counts,
            "splits": splits,
        },
    )

    # Within each few-shot seed, k-shot sets are nested: 1 ⊂ 2 ⊂ 4 ⊂ 8 ⊂ 16.
    for seed in FEW_SHOT_SEEDS:
        sample_rng = random.Random(seed)
        ordered_train_by_class = {}
        for class_id in range(len(classes)):
            candidates = split_by_class[class_id]["train_pool"].copy()
            sample_rng.shuffle(candidates)
            ordered_train_by_class[class_id] = candidates
        for shots in SHOT_COUNTS:
            selected = sorted(
                index
                for class_id in range(len(classes))
                for index in ordered_train_by_class[class_id][:shots]
            )
            assert len(selected) == shots * len(classes)
            assert set(selected).issubset(train_set)
            save_json(
                MANIFEST_DIR / f"eurosat_shot_{shots}_seed_{seed}.json",
                {
                    **common,
                    "source_split_file": "eurosat_split.json",
                    "few_shot_seed": seed,
                    "shots_per_class": shots,
                    "indices": selected,
                },
            )

    print("\nSplit summary:")
    for split_name, indices in splits.items():
        print(f"  {split_name}: {len(indices):,} images")
    print(f"  classes: {len(classes)}; dataset total: {len(dataset):,}")
    print(f"  few-shot manifests: {len(FEW_SHOT_SEEDS) * len(SHOT_COUNTS)}")
    print("Success: disjoint splits and few-shot manifests created.")
    print("Note: validation labels are separate from the few-shot training budget.")


if __name__ == "__main__":
    main()
