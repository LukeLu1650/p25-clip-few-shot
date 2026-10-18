"""Run one dataset/method/shot/seed, selecting settings on validation only."""
import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p25.data import Images, TEMPLATES, digest, import_team_eurosat, prepare, validate_manifest


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=list(TEMPLATES), required=True)
    p.add_argument("--method", choices=["zero_shot", "coop", "tip_adapter"], required=True)
    p.add_argument("--shots", type=int, choices=[1, 2, 4, 8, 16], default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--manifest-dir", type=Path, default=Path("manifests"))
    p.add_argument("--eurosat-manifests", type=Path, default=Path("data/manifests"))
    p.add_argument("--output-root", type=Path, default=Path("results/raw"))
    p.add_argument("--feature-cache", type=Path, default=Path(".cache/features"))
    p.add_argument("--clip-cache", type=Path, default=Path(os.environ.get("P25_CLIP_CACHE", ".cache/clip")))
    p.add_argument("--download", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=.002)
    p.add_argument("--n-ctx", type=int, default=16)
    p.add_argument("--ctx-init", default="")
    p.add_argument("--template", default=None)
    p.add_argument("--val-shots", type=int, default=4)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--resolutions", nargs="+", type=int, default=[0], help="0=native; EuroSAT robustness: 0 16 24 32 64")
    p.add_argument("--alphas", nargs="+", type=float, default=[0, .1, 1, 2, 4, 8])
    p.add_argument("--betas", nargs="+", type=float, default=[.5, 1, 2, 4, 8, 16])
    p.add_argument("--overwrite", action="store_true", help="Explicitly replace a completed identical configuration")
    return p.parse_args()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def main():
    args = arguments()
    if args.batch_size < 1 or args.val_shots < 1 or any(r < 0 for r in args.resolutions):
        raise ValueError("Invalid batch size, validation budget, or resolution")
    args.resolutions = list(dict.fromkeys(args.resolutions))
    if args.dataset == 'eurosat':
        source = args.eurosat_manifests / 'eurosat_split.json'
        if not source.exists():
            raise FileNotFoundError('First run: python src/make_eurosat_splits.py (uses teammate shared splits)')
        args.split_seed = json.loads(source.read_text(encoding='utf-8'))['split_seed']
        args.val_shots = 0  # Explicitly count the full teammate validation partition.
        manifest = import_team_eurosat(args.data_root, args.eurosat_manifests, args.shots, args.seed)
    manifest_path = args.manifest_dir / args.dataset / f"split{args.split_seed}_val{args.val_shots}_k{args.shots}_seed{args.seed}.json"
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.dataset == 'eurosat' and stored != manifest:
            raise ValueError('Stored manifest differs from teammate manifests; coordinate before replacing')
        manifest = stored
        validate_manifest(manifest)
        expected = (args.dataset, args.shots, args.seed, args.val_shots, args.split_seed)
        actual = tuple(manifest[k] for k in ("dataset", "shots", "seed", "val_shots", "split_seed"))
        if actual != expected:
            raise ValueError("Manifest metadata does not match command")
    else:
        if args.dataset != 'eurosat':
            manifest = prepare(args.dataset, args.data_root, args.shots, args.seed,
                               args.download, args.val_shots, args.split_seed)
        write_json(manifest_path, manifest)
    print(f"Manifest: {manifest_path}; {manifest['protocol']}", flush=True)
    if args.prepare_only:
        return
    import clip
    import numpy as np
    import torch
    import torchvision
    from PIL import Image
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from p25.methods import CoOp, accuracy, text_weights, tip_logits, train_coop, tune_tip

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu explicitly")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ("overwrite", "download", "prepare_only", "output_root")}
    config["manifest_sha256"] = digest(manifest)
    config["template"] = args.template or TEMPLATES[args.dataset]
    run_id = digest(config)[:12]
    out = args.output_root / args.dataset / args.method / f"k{args.shots}_seed{args.seed}_{run_id}"
    if (out / "metrics.json").exists() and not args.overwrite:
        print(f"Completed run exists: {out}. Skipping.")
        return
    out.mkdir(parents=True, exist_ok=True)

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    start = time.perf_counter()
    model, preprocess = clip.load("ViT-B/32", device=args.device, jit=False, download_root=str(args.clip_cache))
    model.float().eval()
    model.requires_grad_(False)
    sync()
    load_seconds = time.perf_counter() - start

    def features(rows, resolution=0):
        # Cache identity includes ordering, preprocessing and software. Data is assumed immutable.
        identity = dict(rows=rows, root=str(args.data_root.resolve()), resolution=resolution,
                        model="OpenAI/ViT-B/32:40d365715913c9da", dtype="fp32",
                        torch=torch.__version__, torchvision=torchvision.__version__, schema=1)
        cache = args.feature_cache / (digest(identity) + ".pt")
        tick = time.perf_counter()
        if cache.exists():
            obj = torch.load(cache, map_location=args.device, weights_only=True)
            sync()
            return (obj["features"], obj["labels"]), {"cache_hit": True, "seconds": time.perf_counter()-tick,
                "original_extraction_seconds": obj["extraction_seconds"]}
        def transform(image):
            if resolution:
                image = image.resize((resolution, resolution), Image.Resampling.BICUBIC)
            return preprocess(image)
        loader = DataLoader(Images(rows, args.data_root, transform), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
        xs, ys = [], []
        with torch.no_grad():
            for batch_index, (images, labels) in enumerate(loader):
                xs.append(F.normalize(model.encode_image(images.to(args.device)).float(), dim=-1))
                ys.append(labels.to(args.device))
                if (batch_index + 1) % 50 == 0:
                    print(f'Encoded {min((batch_index+1)*args.batch_size, len(rows))}/{len(rows)} images (r={resolution})', flush=True)
        x, y = torch.cat(xs), torch.cat(ys)
        sync()
        seconds = time.perf_counter() - tick
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": x.cpu(), "labels": y.cpu(), "extraction_seconds": seconds}, cache)
        return (x, y), {"cache_hit": False, "seconds": seconds, "original_extraction_seconds": seconds}

    sync()
    start = time.perf_counter()
    text = text_weights(model, manifest["classes"], config["template"])
    sync()
    text_seconds = time.perf_counter() - start
    timing, selected, trainable = {}, {}, 0
    if args.method != "zero_shot":
        train, timing["train"] = features(manifest["train"])
        val, timing["val"] = features(manifest["val"])
    sync()
    start = time.perf_counter()
    if args.method == "tip_adapter":
        keys = train[0]
        values = F.one_hot(train[1], len(text)).float()
        selected = tune_tip(*val, text, keys, values, args.alphas, args.betas)
        def predict(x):
            return tip_logits(x, text, keys, values, selected["alpha"], selected["beta"])
        torch.save({"keys": keys.cpu(), "values": values.cpu(), "selected": selected}, out / "adapter.pt")
    elif args.method == "coop":
        prompt = CoOp(model, manifest["classes"], args.n_ctx, args.ctx_init)
        selected = train_coop(prompt, train, val, args.epochs, args.batch_size, args.lr, args.seed)
        with torch.no_grad():
            text = prompt.text_features().detach()
        trainable = prompt.context.numel()
        torch.save({"context": prompt.context.detach().cpu(), "classes": manifest["classes"],
                    "n_ctx": args.n_ctx, "ctx_init": args.ctx_init}, out / "prompt.pt")
        def predict(x):
            return model.logit_scale.exp() * x @ text.T
    else:
        def predict(x):
            return model.logit_scale.exp() * x @ text.T
    sync()
    optimization_seconds = time.perf_counter() - start
    scores, eval_times = {}, {}
    # Test labels are used only after adaptation and hyperparameter selection are finished.
    for resolution in args.resolutions:
        (x, y), timing[f"test_{resolution}"] = features(manifest["test"], resolution)
        sync()
        tick = time.perf_counter()
        with torch.no_grad():
            logits = torch.cat([predict(part) for part in x.split(args.batch_size)])
        sync()
        eval_times[str(resolution)] = time.perf_counter() - tick
        scores[str(resolution)] = accuracy(logits, y)
        pred = logits.argmax(-1).cpu().tolist()
        with (out / f"predictions_r{resolution}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["path", "label", "prediction"])
            writer.writerows((r["path"], r["label"], p) for r, p in zip(manifest["test"], pred))
        confusion = torch.bincount(y * len(text) + logits.argmax(-1), minlength=len(text)**2).reshape(len(text), len(text))
        write_json(out / f"confusion_r{resolution}.json", {"classes": manifest["classes"], "rows_true_columns_pred": confusion.cpu().tolist()})
        # Text-to-image retrieval is qualitative; rankings use encoder similarity, not cache logits.
        ranks = (x @ text.T).topk(min(5, len(x)), dim=0).indices.cpu().T.tolist()
        write_json(out / f"retrieval_r{resolution}.json", {c: [manifest["test"][i] for i in ids] for c, ids in zip(manifest["classes"], ranks)})
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    adaptation_keys = [k for k in ("train", "val") if k in timing]
    result = dict(status="complete", timestamp_utc=datetime.now(timezone.utc).isoformat(),
        config=config, protocol=manifest["protocol"], pool_sha256=manifest["pool_sha256"],
        accuracy=scores, trainable_parameters=trainable, selected=selected,
        label_budget={"train": len(manifest["train"]) if args.method != "zero_shot" else 0,
                      "validation": len(manifest["val"]) if args.method != "zero_shot" else 0},
        timing={"model_load_seconds": load_seconds, "text_encoding_seconds": text_seconds,
                "feature_extraction": timing, "optimization_and_selection_seconds": optimization_seconds,
                "evaluation_seconds": eval_times,
                "cold_adaptation_seconds": text_seconds + optimization_seconds + sum(timing[k]["original_extraction_seconds"] for k in adaptation_keys)},
        environment={"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                     "hardware": torch.cuda.get_device_name(0) if args.device == "cuda" else platform.processor(),
                     "clip_commit": "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6", "git_revision": revision},
        caveat="Simplified cached-feature baselines; see docs/PROTOCOL.md. No claim of published-number reproduction.")
    write_json(out / "metrics.json", result)
    print(json.dumps({"output": str(out), "accuracy": scores}, indent=2), flush=True)


if __name__ == "__main__":
    main()
