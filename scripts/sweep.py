"""One command for the baseline matrix or controlled CoOp sensitivity runs."""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baselines.json"))
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    jobs = []
    for dataset in cfg["datasets"]:
        for method in cfg["methods"]:
            # Zero-shot has no sample/training seed and is evaluated once per dataset.
            pairs = [(1, cfg["seeds"][0])] if method == "zero_shot" else itertools.product(cfg["shots"], cfg["seeds"])
            for shots, seed in pairs:
                variants = cfg.get("coop_variants", [{}]) if method == "coop" else [{}]
                for variant in variants:
                    command = [sys.executable, str(Path(__file__).with_name("run_experiment.py")),
                               "--dataset", dataset, "--method", method, "--shots", str(shots), "--seed", str(seed)]
                    for key, value in {**cfg.get("options", {}), **variant}.items():
                        command.append("--" + key.replace("_", "-"))
                        command.extend(map(str, value if isinstance(value, list) else [value]))
                    jobs.append(command + extra)
    print(f"{len(jobs)} experiments", flush=True)
    for i, command in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {subprocess.list2cmdline(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
