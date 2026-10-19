"""EuroSAT controlled single- versus multi-resolution border prompt DEVELOPMENT runs.

Place at scripts/train_resolution_prompts.py inside p25-clip-few-shot.
Both conditions train the SAME shared pixel-border prompt with fixed CLIP image/text
encoders, matched few-shot lists, optimizer, image order and update budgets.
Only the effective input-resolution strategy differs:
  single: each image is resized to --train-resolution before CLIP preprocessing;
  multi:  each training image independently samples 16, 32, or 64 (uniform).

The same validation-only model-selection score is used for all methods:
    mean(top1_val_at_16, top1_val_at_32, top1_val_at_64).
Never use 24 for training/selection: reserve it for held-out resolution evaluation
AFTER all configurations are frozen. The test split is NEVER accessed.
This runner is for development; do not mix its output with the reviewed 93 baselines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader

from p25.data import Images, TEMPLATES, digest, import_team_eurosat
from p25.methods import text_weights
from p25.pixel_prompt import PixelBorderPrompt

TRAIN_RESOLUTIONS = (16, 32, 64)
SELECT_RESOLUTIONS = (16, 32, 64)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("single", "multi"))
    p.add_argument("--train-resolution", type=int, choices=TRAIN_RESOLUTIONS, default=None,
                   help="Required for --mode single; ignored/forbidden for multi")
    p.add_argument("--shots", type=int, choices=(1, 2, 4, 8, 16), default=16)
    p.add_argument("--seed", type=int, choices=(1, 2, 3), default=1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=0, help="0=full epochs; for smoke tests only")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--val-batch-size", type=int, default=8)
    p.add_argument("--val-per-class", type=int, default=0,
                   help="0=all 2700 validation images; positive=DEVELOPMENT subset only")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--border-width", type=int, default=8)
    p.add_argument("--max-delta", type=float, default=0.25)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--eurosat-manifests", type=Path, default=ROOT / "data" / "manifests")
    p.add_argument("--clip-cache", type=Path, default=ROOT / ".cache" / "clip")
    p.add_argument("--output-root", type=Path, default=ROOT / "results" / "resolution_prompt_dev")
    args = p.parse_args()
    if args.mode == "single" and args.train_resolution is None:
        p.error("--mode single requires --train-resolution 16, 32, or 64")
    if args.mode == "multi" and args.train_resolution is not None:
        p.error("--mode multi must not set --train-resolution")
    if (args.epochs < 1 or args.max_steps < 0 or args.batch_size < 1 or
            args.val_batch_size < 1 or args.val_per_class < 0 or args.lr <= 0 or
            args.border_width < 1 or not 0 < args.max_delta <= 1):
        p.error("Invalid optimization/validation/prompt setting")
    return args


def make_transform(preprocess, *, mode, resolution=None, rng=None):
    """The effective resize happens BEFORE the same official CLIP preprocessing.

    Using a local RNG in mode=multi makes sampling reproducible and independent
    of Python global random state. num_workers=0 keeps sampling order stable.
    """
    if mode not in ("single", "multi"):
        raise ValueError("mode must be single or multi")
    if mode == "single" and resolution not in TRAIN_RESOLUTIONS:
        raise ValueError("single transform requires a valid resolution")
    if mode == "multi" and rng is None:
        raise ValueError("multi transform requires a seeded rng")

    def transform(image):
        r = resolution if mode == "single" else rng.choice(TRAIN_RESOLUTIONS)
        return preprocess(image.resize((r, r), resample=Image.Resampling.BICUBIC))
    return transform


def choose_validation_subset(rows, per_class):
    if per_class == 0:
        return rows
    by_class = defaultdict(list)
    for row in rows:
        by_class[row["label"]].append(row)
    if len(by_class) != 10 or any(len(v) < per_class for v in by_class.values()):
        raise ValueError("Cannot select requested validation subset")
    return [item for label in sorted(by_class) for item in by_class[label][:per_class]]


def get_logits(model, prompt, images, text):
    # Do NOT use torch.no_grad here during training: pixel prompt requires
    # backpropagation through the frozen CLIP image encoder to the input image.
    features = F.normalize(model.encode_image(prompt(images)).float(), dim=-1)
    return model.logit_scale.exp().detach() * (features @ text.T)


@torch.no_grad()
def validate(model, prompt, loaders, text, device):
    prompt.eval()
    scores = {}
    for resolution, loader in loaders.items():
        correct, count = 0, 0
        for images, labels in loader:
            logits = get_logits(model, prompt, images.to(device), text)
            correct += (logits.argmax(-1).cpu() == labels).sum().item()
            count += len(labels)
        if count == 0:
            raise ValueError("Empty validation set")
        scores[str(resolution)] = correct / count * 100.0
    return scores


def mean_score(scores):
    if tuple(sorted(map(int, scores))) != SELECT_RESOLUTIONS:
        raise ValueError("Selection must use 16, 32 and 64 only")
    return sum(scores[str(r)] for r in SELECT_RESOLUTIONS) / len(SELECT_RESOLUTIONS)


def state_on_cpu(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: expected GPU-enabled local environment")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda"

    manifest = import_team_eurosat(args.data_root, args.eurosat_manifests, args.shots, args.seed)
    if len(manifest["train"]) != 10 * args.shots or len(manifest["val"]) != 2700:
        raise ValueError("Unexpected EuroSAT shared train/validation sizes")
    subset_rows = choose_validation_subset(manifest["val"], args.val_per_class)
    if args.val_per_class == 0 and len(subset_rows) != 2700:
        raise ValueError("Full validation expected")

    import clip  # OFFICIAL OpenAI CLIP, matching the team's baselines
    started = time.perf_counter()
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False,
                                  download_root=str(args.clip_cache))
    model.float().eval().requires_grad_(False)
    text = text_weights(model, manifest["classes"], TEMPLATES["eurosat"]).detach()
    prompt = PixelBorderPrompt(image_size=224, border_width=args.border_width,
                               max_delta=args.max_delta).to(device)
    optimizer = torch.optim.AdamW([prompt.raw_delta], lr=args.lr, weight_decay=0.0)

    # Identical image order in all conditions; only transform policy differs.
    train_rng = random.Random(91800 + args.seed)
    train_transform = make_transform(
        preprocess, mode=args.mode, resolution=args.train_resolution, rng=train_rng)
    train_loader = DataLoader(
        Images(manifest["train"], args.data_root, train_transform),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed))
    validation_loaders = {
        r: DataLoader(
            Images(subset_rows, args.data_root,
                   make_transform(preprocess, mode="single", resolution=r)),
            batch_size=args.val_batch_size, shuffle=False, num_workers=0)
        for r in SELECT_RESOLUTIONS
    }

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                          text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
    config = {
        "method": f"{args.mode}_resolution_pixel_border_prompt_DEVELOPMENT",
        "dataset": "eurosat", "shots": args.shots, "seed": args.seed,
        "mode": args.mode,
        "train_resolution": args.train_resolution if args.mode == "single" else None,
        "train_resolution_choices": list(TRAIN_RESOLUTIONS) if args.mode == "multi" else [args.train_resolution],
        "train_resolution_sampling": "uniform per image, seeded" if args.mode == "multi" else "fixed",
        "selection_resolutions": list(SELECT_RESOLUTIONS),
        "checkpoint_rule": "highest mean val top-1 across 16,32,64; earliest epoch on ties; epoch 0 included",
        "epochs": args.epochs, "max_steps": args.max_steps,
        "batch_size": args.batch_size, "val_batch_size": args.val_batch_size,
        "val_per_class": args.val_per_class,
        "lr": args.lr, "optimizer": "AdamW(weight_decay=0)",
        "border_width": args.border_width, "max_delta": args.max_delta,
        "precision": "fp32", "model": "OpenAI CLIP ViT-B/32",
        "template": TEMPLATES["eurosat"], "protocol": manifest["protocol"],
        "pool_sha256": manifest["pool_sha256"],
        "manifest_sha256": digest(manifest),
        "validation_labels_available": len(manifest["val"]),
        "validation_labels_evaluated": len(subset_rows),
        "unseen_resolution_24_used": False,
        "test_partition_evaluated": False,
        "git_revision": git_sha,
    }
    sig = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    name = f"k{args.shots}_s{args.seed}_{args.mode}"
    if args.mode == "single":
        name += str(args.train_resolution)
    run_dir = args.output_root / f"{name}_{sig}"
    if run_dir.exists():
        raise FileExistsError(f"Existing run; will not overwrite: {run_dir}")
    print(f"mode={args.mode}; train_images={len(manifest['train'])}; "
          f"validation_images={len(subset_rows)}; selection_resolutions={SELECT_RESOLUTIONS}; "
          "test/24x24 NOT USED", flush=True)

    training_seconds = 0.0
    validation_seconds = 0.0

    def evaluate_timed(module):
        nonlocal validation_seconds
        torch.cuda.synchronize()
        tick = time.perf_counter()
        values = validate(model, module, validation_loaders, text, device)
        torch.cuda.synchronize()
        validation_seconds += time.perf_counter() - tick
        return values

    initial = evaluate_timed(prompt)
    best_score = mean_score(initial)
    best_epoch = 0
    best_state = state_on_cpu(prompt)
    history = [dict(epoch=0, updates=0, mean_train_loss=None,
                    val_accuracy=initial, selection_score=best_score)]
    print(f"Epoch 0: val={initial}; selection_mean={best_score:.3f}%", flush=True)

    steps = 0
    for epoch in range(1, args.epochs + 1):
        prompt.train()
        samples = 0
        loss_weighted = 0.0
        updates = 0
        torch.cuda.synchronize()
        tick = time.perf_counter()
        for images, labels in train_loader:
            if args.max_steps and steps >= args.max_steps:
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(get_logits(model, prompt, images, text), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            grad = prompt.raw_delta.grad
            if (grad is None or not torch.isfinite(grad).all() or
                    (grad * prompt.mask).abs().sum().item() == 0):
                raise RuntimeError("Missing, nonfinite or zero border gradient")
            optimizer.step()
            steps += 1
            updates += 1
            samples += len(labels)
            loss_weighted += loss.item() * len(labels)
            if steps % 20 == 0:
                print(f"Update {steps}: loss={loss.item():.4f}", flush=True)
        torch.cuda.synchronize()
        training_seconds += time.perf_counter() - tick
        if updates == 0:
            break
        val = evaluate_timed(prompt)
        score = mean_score(val)
        mean_loss = loss_weighted / samples
        history.append(dict(epoch=epoch, updates=steps, mean_train_loss=mean_loss,
                            val_accuracy=val, selection_score=score))
        print(f"Epoch {epoch}/{args.epochs}: loss={mean_loss:.4f}, "
              f"val={val}, mean={score:.3f}%, steps={steps}", flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = state_on_cpu(prompt)
        if args.max_steps and steps >= args.max_steps:
            break

    # Save, then verify by actually reloading the file (not merely an in-memory state).
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = run_dir / "best_prompt.pt"
    torch.save({"state_dict": best_state, "config": config,
                "best_epoch": best_epoch, "best_selection_score": best_score}, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    restored = PixelBorderPrompt(image_size=224, border_width=args.border_width,
                                 max_delta=args.max_delta).to(device)
    restored.load_state_dict(loaded["state_dict"])
    reloaded_val = evaluate_timed(restored)
    reloaded_score = mean_score(reloaded_val)
    if not math.isclose(reloaded_score, best_score, rel_tol=0, abs_tol=1e-5):
        raise RuntimeError("Reloaded checkpoint does not reproduce selection score")
    if best_epoch > 0 and torch.count_nonzero(restored.raw_delta).item() == 0:
        raise RuntimeError("Selected trained prompt has no nonzero learned parameters")

    output = {
        "status": "development_validation_only",
        "config": config,
        "optimizer_steps": steps,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "best_validation_accuracy_by_resolution": reloaded_val,
        "reload_selection_score": reloaded_score,
        "history": history,
        "effective_trainable_parameters": restored.effective_trainable_parameters(),
        "timing_seconds": {"training": training_seconds,
                           "validation_including_reload": validation_seconds,
                           "total_including_model_load": time.perf_counter() - started},
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0)},
        "unseen_resolution_24_used": False,
        "test_partition_evaluated": False,
        "warning": "Development validation only; do not merge into baseline summary.csv or claim final robustness.",
    }
    (run_dir / "training.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Success: best_epoch={best_epoch}, mean={best_score:.3f}%, "
          f"reloaded_mean={reloaded_score:.3f}%; saved in {run_dir}", flush=True)
    print("Test/24x24 never evaluated.", flush=True)


if __name__ == "__main__":
    main()
