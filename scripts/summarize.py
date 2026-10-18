"""Aggregate independent seeds, rejecting duplicate runs within a condition."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(root):
    groups = defaultdict(dict)
    for path in sorted(Path(root).rglob("metrics.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            continue
        c = result["config"]
        # Include all scientific settings, pool and validation protocol in the group.
        ignored = {"seed", "manifest_sha256", "data_root", "manifest_dir", "feature_cache", "clip_cache", "device", "resolutions"}
        condition = json.dumps({k: v for k, v in c.items() if k not in ignored}, sort_keys=True)
        for resolution, value in result["accuracy"].items():
            key = (condition, result["pool_sha256"], resolution)
            if c["seed"] in groups[key]:
                raise ValueError(f"Duplicate seed {c['seed']} in condition: {path}. Choose one results folder.")
            groups[key][c["seed"]] = (value, result)
    rows = []
    for (condition, pool, resolution), seeds in sorted(groups.items()):
        c = json.loads(condition)
        values = [v[0] for v in seeds.values()]
        rows.append({"dataset": c["dataset"], "method": c["method"], "shots": c["shots"],
                     "resolution": resolution, "n": len(values), "seeds": ";".join(map(str, sorted(seeds))),
                     "mean_accuracy": statistics.mean(values),
                     "sample_std": statistics.stdev(values) if len(values) > 1 else "",
                     "sufficient_seeds": c["method"] == "zero_shot" or len(values) >= 3,
                     "trainable_parameters": next(iter(seeds.values()))[1]["trainable_parameters"],
                     "mean_cold_adaptation_seconds": statistics.mean(v[1]["timing"]["cold_adaptation_seconds"] for v in seeds.values()),
                     "pool_sha256": pool, "condition": condition})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/raw"))
    parser.add_argument("--output", type=Path, default=Path("results/summary.csv"))
    args = parser.parse_args()
    rows = summarize(args.root)
    if not rows:
        raise SystemExit("No completed experiments found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} conditions to {args.output}")


if __name__ == "__main__":
    main()
