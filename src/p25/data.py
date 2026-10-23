"""Fixed partitions and portable manifests shared by every method."""
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError

TEMPLATES = {
    "eurosat": "a satellite image of {}.",
    "dtd": "a photo of a {} texture.",
    "flowers102": "a photo of a {}, a type of flower.",
}
EUROSAT_NAMES = {
    "AnnualCrop": "annual crop land", "Forest": "forest",
    "HerbaceousVegetation": "herbaceous vegetation land", "Highway": "highway or road",
    "Industrial": "industrial buildings", "Pasture": "pasture land",
    "PermanentCrop": "permanent crop land", "Residential": "residential buildings",
    "River": "river", "SeaLake": "sea or lake",
}


def load_with_download_retry(factory, download, attempts=3):
    """Retry temporary download failures; leave permanent HTTP errors visible."""
    if attempts < 1:
        raise ValueError('Download attempts must be positive')
    for attempt in range(1, attempts + 1):
        try:
            return factory()
        except (URLError, TimeoutError) as error:
            transient_http = not isinstance(error, HTTPError) or error.code in (408, 429, 500, 502, 503, 504)
            if not download or not transient_http or attempt == attempts:
                raise
            delay = 2 ** attempt
            print(f'Dataset download temporarily failed: {error}. Retrying in {delay}s '
                  f'(attempt {attempt + 1}/{attempts}).', flush=True)
            time.sleep(delay)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def stratify(rows, split_seed=0):
    """50/20/30 per class; split RNG is independent of few-shot training RNG."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    result = {s: [] for s in ("train", "val", "test")}
    rng = random.Random(split_seed)
    for label in sorted(groups):
        items = sorted(groups[label], key=lambda r: r["path"])
        rng.shuffle(items)
        a, b = round(len(items) * .5), round(len(items) * .2)
        if min(a, b, len(items) - a - b) < 1:
            raise ValueError("Each class needs nonempty train/val/test partitions")
        for split, selected in zip(result, (items[:a], items[a:a+b], items[a+b:])):
            result[split].extend(selected)
    return result


def sample(rows, shots, seed):
    if shots < 1:
        raise ValueError("Shots must be positive")
    groups = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    selected, rng = [], random.Random(seed)
    for label in sorted(groups):
        items = sorted(groups[label], key=lambda r: r["path"])
        if len(items) < shots:
            raise ValueError(f"Class {label}: requested {shots} but only {len(items)} available")
        rng.shuffle(items)
        selected.extend(items[:shots])
    return selected


def validate_manifest(manifest):
    classes = manifest["classes"]
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("Class names must be nonempty and unique")
    seen = set()
    for split in ("train", "val", "test"):
        rows = manifest[split]
        if set(r["label"] for r in rows) != set(range(len(classes))):
            raise ValueError(f"{split} must contain every class")
        for row in rows:
            p = Path(row["path"])
            if p.is_absolute() or ".." in p.parts or row["path"] in seen:
                raise ValueError(f"Unsafe or overlapping image path: {row['path']}")
            seen.add(row["path"])
    for split in ("train", "val"):
        expected = manifest["shots"] if split == "train" else manifest["val_shots"]
        if expected and set(Counter(r["label"] for r in manifest[split]).values()) != {expected}:
            raise ValueError(f"Incorrect {split} label budget")


def import_team_eurosat(root, manifest_dir, shots, seed):
    """Import shared index manifests and verify dataset ordering."""
    from torchvision.datasets import EuroSAT
    root, manifest_dir = Path(root).resolve(), Path(manifest_dir)
    split = json.loads((manifest_dir / 'eurosat_split.json').read_text(encoding='utf-8'))
    few = json.loads((manifest_dir / f'eurosat_shot_{shots}_seed_{seed}.json').read_text(encoding='utf-8'))
    ds = EuroSAT(root=str(root), download=False)
    fingerprint = hashlib.sha256()
    for i, (path, label) in enumerate(ds.samples):
        relative = Path(path).relative_to(Path(ds.root)).as_posix()
        fingerprint.update(f'{i}\t{relative}\t{label}\n'.encode())
    actual = fingerprint.hexdigest()
    for obj in (split, few):
        if obj['dataset_fingerprint_sha256'] != actual or obj['classes'] != list(ds.classes) or obj['dataset_size'] != len(ds):
            raise ValueError('Teammate manifest dataset fingerprint/class ordering mismatch')
    if few['shots_per_class'] != shots or few['few_shot_seed'] != seed or few['split_seed'] != split['split_seed']:
        raise ValueError('Teammate few-shot metadata mismatch')
    pool, val, test = (split['splits'][s] for s in ('train_pool', 'val', 'test'))
    if any(len(v) != len(set(v)) for v in (pool, val, test, few['indices'])):
        raise ValueError('Duplicate indices in teammate manifests')
    if set(pool) & set(val) or set(pool) & set(test) or set(val) & set(test):
        raise ValueError('Teammate train/validation/test overlap')
    if set(pool) | set(val) | set(test) != set(range(len(ds))):
        raise ValueError('Teammate partitions do not cover the dataset exactly')
    if not set(few['indices']) <= set(pool):
        raise ValueError('Few-shot indices are outside teammate train pool')
    def rows(indices):
        return [{'path': Path(ds.samples[i][0]).resolve().relative_to(root).as_posix(),
                 'label': int(ds.samples[i][1])} for i in indices]
    result = dict(dataset='eurosat', classes=[EUROSAT_NAMES[c] for c in ds.classes],
                  shots=shots, seed=seed, val_shots=0, split_seed=split['split_seed'],
                  protocol=f"Teammate EuroSAT RGB 60/10/30; split_seed={split['split_seed']}; ALL validation labels",
                  pool_sha256=digest(split), schema_version=1, source_fingerprint=actual,
                  train=rows(few['indices']), val=rows(val), test=rows(test))
    validate_manifest(result)
    return result


def prepare(dataset, root, shots, seed, download=False, val_shots=4, split_seed=0):
    from torchvision.datasets import DTD, EuroSAT, Flowers102
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    def row(path, label):
        return {"path": Path(path).resolve().relative_to(root).as_posix(), "label": int(label)}

    if dataset == "eurosat":
        ds = load_with_download_retry(lambda: EuroSAT(root=str(root), download=download), download)
        classes = [EUROSAT_NAMES[c] for c in ds.classes]
        pools = stratify([row(p, y) for p, y in ds.samples], split_seed)
        protocol = "EuroSAT RGB stratified 50/20/30, split_seed=" + str(split_seed)
    elif dataset == "dtd":
        pools = {}
        for split in ("train", "val", "test"):
            ds = load_with_download_retry(
                lambda: DTD(root=str(root), split=split, partition=1, download=download), download)
            classes = ds.classes
            pools[split] = [row(p, y) for p, y in zip(ds._image_files, ds._labels)]
        protocol = "DTD official partition 1 (40/40/40 images per class)"
    elif dataset == "flowers102":
        rows = []
        for split in ("train", "val", "test"):
            ds = load_with_download_retry(
                lambda: Flowers102(root=str(root), split=split, download=download), download)
            rows.extend(row(p, y) for p, y in zip(ds._image_files, ds._labels))
        classes = ds.classes
        pools = stratify(rows, split_seed)
        protocol = "Flowers102 custom stratified 50/20/30 over ALL images, split_seed=" + str(split_seed)
    else:
        raise ValueError(dataset)
    result = dict(dataset=dataset, classes=classes, shots=shots, seed=seed,
                  val_shots=val_shots, split_seed=split_seed, protocol=protocol,
                  pool_sha256=digest(pools), schema_version=1,
                  train=sample(pools["train"], shots, seed),
                  val=sample(pools["val"], val_shots, split_seed), test=pools["test"])
    validate_manifest(result)
    return result


class Images:
    def __init__(self, rows, root, transform):
        self.rows, self.root, self.transform = rows, Path(root), transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        row = self.rows[index]
        with Image.open(self.root / row["path"]) as image:
            value = self.transform(image.convert("RGB"))
        return value, row["label"]
