"""Predeclared zero-shot DTD check against CLIP Table 11 (ViT-B/32: 44.5%)."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def validate_test_rows(classes, expected_classes, rows, expected_size):
    if list(classes) != list(expected_classes):
        raise ValueError("DTD class ordering differs from the official prompt metadata")
    if len(rows) != expected_size or len({r["path"] for r in rows}) != expected_size:
        raise ValueError("Incorrect test size or duplicate test images")
    counts = Counter(r["label"] for r in rows)
    if set(counts) != set(range(47)) or set(counts.values()) != {40}:
        raise ValueError("Expected exactly 40 test images in each of 47 classes")


def ensemble_text_features(model, classes, templates, tokenize):
    """Normalize each prompt, average within class, then normalize again."""
    import torch
    from torch.nn import functional as F

    device = next(model.parameters()).device
    weights = []
    with torch.no_grad():
        for name in classes:
            tokens = tokenize([t.format(name) for t in templates]).to(device)
            embeddings = F.normalize(model.encode_text(tokens), dim=-1)
            weights.append(F.normalize(embeddings.mean(dim=0), dim=0))
    return torch.stack(weights, dim=1)


def compare_accuracy(accuracy, published):
    gap = abs(accuracy - published)
    return {
        "published_accuracy_percent": published,
        "absolute_gap_percentage_points": gap,
        "relative_error_percent": 100 * gap / published,
        "within_1_percentage_point": gap <= 1,
        "within_2_percentage_points": gap <= 2,
        "within_2_percent_relative": gap <= .02 * published,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "configs/reproduction_clip_dtd.json")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "results/reproduction/clip_dtd_vit_b32")
    parser.add_argument("--clip-cache", type=Path,
                        default=Path(os.environ.get("P25_CLIP_CACHE", PROJECT_ROOT / ".cache/clip")))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected = ("dtd", "ViT-B/32", "test", 1, 1880, 44.5)
    actual = tuple(config[k] for k in (
        "dataset", "backbone", "split", "partition", "test_size", "published_accuracy_percent"))
    if actual != expected or len(config["classes"]) != 47 or len(config["templates"]) != 8:
        raise ValueError("Config does not describe the predeclared DTD Table 11 check")
    if (args.output_dir / "metrics.json").exists():
        raise FileExistsError("Completed result exists; use a different --output-dir for a new run")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import clip
    import torch
    import torchvision
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torchvision.datasets import DTD

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select --device cpu explicitly")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Keep the official loader's native precision: CUDA FP16 / CPU FP32.
    model, preprocess = clip.load(config["backbone"], device=args.device,
                                  jit=False, download_root=str(args.clip_cache))
    model.eval().requires_grad_(False)
    checkpoint = args.clip_cache / "ViT-B-32.pt"
    if sha256(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match the official ViT-B/32 weights")
    dataset = DTD(root=str(args.data_root), split="test", partition=1,
                  transform=preprocess, download=args.download)
    rows = [{"path": p.resolve().relative_to(args.data_root.resolve()).as_posix(),
             "label": int(y)} for p, y in zip(dataset._image_files, dataset._labels)]
    validate_test_rows(dataset.classes, config["classes"], rows, config["test_size"])
    split_path = dataset._meta_folder / "test1.txt"
    official_names = split_path.read_text().splitlines()
    actual_names = [p.relative_to(dataset._images_folder).as_posix() for p in dataset._image_files]
    if official_names != actual_names:
        raise ValueError("Image order differs from official test1.txt")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Predeclared target: CLIP Table 11 / DTD / ViT-B/32 / 44.5%", flush=True)
    print("Fixed protocol: test1.txt (1,880 images), eight official prompts; no adaptation", flush=True)
    start = time.perf_counter()
    text = ensemble_text_features(model, dataset.classes, config["templates"], clip.tokenize)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    correct, total, all_predictions = 0, 0, []
    confusion = torch.zeros(47, 47, dtype=torch.int64)
    with torch.no_grad():
        for images, labels in loader:
            features = F.normalize(model.encode_image(images.to(args.device)), dim=-1)
            predictions = (100.0 * features @ text).argmax(dim=-1).cpu()
            correct += int((predictions == labels).sum())
            total += len(labels)
            all_predictions.extend(predictions.tolist())
            confusion += torch.bincount(labels * 47 + predictions, minlength=47**2).reshape(47, 47)
            if total % (10 * args.batch_size) == 0 or total == len(dataset):
                print(f"Evaluated {total}/{len(dataset)}", flush=True)
    accuracy = 100.0 * correct / total
    comparison = compare_accuracy(accuracy, config["published_accuracy_percent"])
    metric = {
        "status": "complete", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "config_sha256": sha256(args.config),
        "script_sha256": sha256(Path(__file__)),
        "accuracy_percent": accuracy, "correct": correct, "test_size": total,
        "comparison": comparison, "trainable_parameters": 0,
        "label_budget": {"train": 0, "validation": 0},
        "independent_runs": 1, "sample_std": None,
        "split_file_sha256": sha256(split_path),
        "evaluation_seconds_including_text_encoding": time.perf_counter() - start,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                        "hardware": torch.cuda.get_device_name(0) if args.device == "cuda" else platform.processor(),
                        "model_dtype": str(next(model.parameters()).dtype)},
        "limitations": [
            "Numerical zero-shot validation only; not a reproduction of CoOp or Tip-Adapter training.",
            "Uses official first-fold DTD test list, matching the historical TFDS first-fold convention; the CLIP paper does not explicitly name the DTD fold.",
            "Modern PyTorch/torchvision/Pillow differ from the original software environment.",
            "Fixed zero-shot evaluation has no sampling/training seed variability; do not report three repeated passes as independent runs.",
        ],
    }
    def save_json(name, obj):
        (args.output_dir / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    save_json("test_manifest.json", {"classes": dataset.classes, "test": rows,
                                    "test1_sha256": sha256(split_path)})
    save_json("confusion.json", {"classes": dataset.classes,
                                "rows_true_columns_pred": confusion.tolist()})
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "label", "prediction"])
        writer.writerows((r["path"], r["label"], p) for r, p in zip(rows, all_predictions))
    summary = {"dataset": "dtd", "method": "zero_shot_official_prompt_ensemble",
               "backbone": "ViT-B/32", "n": 1, "accuracy_percent": accuracy, **comparison}
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    save_json("metrics.json", metric)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
