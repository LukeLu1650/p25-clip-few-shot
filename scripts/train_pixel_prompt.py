"""Train a single-resolution border prompt with validation checkpoint selection.

Epoch 0 is eligible; test images are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Configure deterministic CUDA operations before importing torch.
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, choices=(1, 2, 4, 8, 16), default=16)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--train-resolution", type=int, choices=(16, 32, 64), default=64)
    parser.add_argument("--validation-resolution", type=int, choices=(16, 24, 32, 64), default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means no early step limit")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--border-width", type=int, default=8)
    parser.add_argument("--max-delta", type=float, default=0.25)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--eurosat-manifests", type=Path, default=ROOT / "data" / "manifests")
    parser.add_argument("--clip-cache", type=Path, default=ROOT / ".cache" / "clip")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "pixel_prompt_training")
    args = parser.parse_args()
    if (args.epochs < 1 or args.max_steps < 0 or args.batch_size < 1 or
            args.val_batch_size < 1 or args.lr <= 0 or args.border_width < 1 or
            not 0 < args.max_delta <= 1):
        parser.error("Invalid training or prompt setting")
    return args


def make_transform(preprocess, resolution):
    def transform(image):
        image = image.resize((resolution, resolution), Image.Resampling.BICUBIC)
        return preprocess(image)
    return transform


def get_logits(model, prompt, images, text_features):
    # Keep the gradient path from the frozen encoder to the prompt.
    features = F.normalize(model.encode_image(prompt(images)).float(), dim=-1)
    return model.logit_scale.exp().detach() * (features @ text_features.T)


@torch.no_grad()
def evaluate_validation(model, prompt, loader, text_features, device):
    prompt.eval()
    seen = correct = 0
    for images, labels in loader:
        logits = get_logits(model, prompt, images.to(device), text_features)
        correct += (logits.argmax(-1).cpu() == labels).sum().item()
        seen += labels.numel()
    if not seen:
        raise ValueError("Validation loader was empty")
    return 100.0 * correct / seen


def cpu_checkpoint(prompt):
    return {k: value.detach().cpu().clone() for k, value in prompt.state_dict().items()}


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pixel-prompt training.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda"

    manifest = import_team_eurosat(args.data_root, args.eurosat_manifests,
                                   args.shots, args.seed)
    if len(manifest["val"]) != 2700:
        raise ValueError("Expected full 2,700-image EuroSAT validation partition")
    expected_train = 10 * args.shots
    if len(manifest["train"]) != expected_train:
        raise ValueError(f"Expected {expected_train} training images")

    import clip  # Official OpenAI CLIP, matching the team's baseline.
    start_total = time.perf_counter()
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False,
                                  download_root=str(args.clip_cache))
    model.float().eval().requires_grad_(False)
    text_features = text_weights(model, manifest["classes"], TEMPLATES["eurosat"]).detach()
    prompt = PixelBorderPrompt(image_size=224, border_width=args.border_width,
                               max_delta=args.max_delta).to(device)
    optimizer = torch.optim.AdamW([prompt.raw_delta], lr=args.lr, weight_decay=0.0)

    training_loader = DataLoader(
        Images(manifest["train"], args.data_root, make_transform(preprocess, args.train_resolution)),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        Images(manifest["val"], args.data_root, make_transform(preprocess, args.validation_resolution)),
        batch_size=args.val_batch_size, shuffle=False, num_workers=0,
    )

    scientific_config = dict(
        method="standard_pixel_border_prompt_DEVELOPMENT",
        dataset="eurosat", shots=args.shots, seed=args.seed,
        train_resolution=args.train_resolution,
        validation_resolution=args.validation_resolution,
        epochs=args.epochs, max_steps=args.max_steps,
        batch_size=args.batch_size, val_batch_size=args.val_batch_size,
        lr=args.lr, border_width=args.border_width, max_delta=args.max_delta,
        optimizer="AdamW(weight_decay=0)", model="OpenAI CLIP ViT-B/32",
        template=TEMPLATES["eurosat"], precision="fp32",
        data_protocol=manifest["protocol"], pool_sha256=manifest["pool_sha256"],
        manifest_sha256=digest(manifest),
        validation_label_budget=len(manifest["val"]),
        validation_selection_rule="highest accuracy, earliest epoch on ties; epoch 0 included",
        test_partition_used=False,
    )
    signature = hashlib.sha256(json.dumps(scientific_config, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = (args.output_root / f"k{args.shots}_seed{args.seed}_train{args.train_resolution}"
               f"_val{args.validation_resolution}_{signature}")
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {run_dir}")

    print(f"Training: {len(manifest['train'])} images, {args.train_resolution}x{args.train_resolution}; "
          f"validation: {len(manifest['val'])} images, "
          f"{args.validation_resolution}x{args.validation_resolution}; TEST NOT USED", flush=True)
    val_seconds = 0.0
    train_seconds = 0.0
    torch.cuda.synchronize()
    tick = time.perf_counter()
    initial_val = evaluate_validation(model, prompt, validation_loader, text_features, device)
    torch.cuda.synchronize()
    initial_val_seconds = time.perf_counter() - tick
    val_seconds += initial_val_seconds
    best_score = initial_val
    best_epoch = 0
    best_state = cpu_checkpoint(prompt)
    history = [{"epoch": 0, "updates": 0, "val_accuracy": initial_val,
                "mean_train_loss": None, "train_seconds": 0.0,
                "validation_seconds": initial_val_seconds}]
    print(f"Epoch 0 (no prompt training): validation accuracy = {initial_val:.3f}%", flush=True)

    total_steps = 0
    for epoch in range(1, args.epochs + 1):
        prompt.train()
        loss_sum = 0.0
        samples_seen = 0
        updates_in_epoch = 0
        torch.cuda.synchronize()
        tick = time.perf_counter()
        for images, labels in training_loader:
            if args.max_steps and total_steps >= args.max_steps:
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(get_logits(model, prompt, images, text_features), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            grad = prompt.raw_delta.grad
            if (grad is None or not torch.isfinite(grad).all() or
                    (grad * prompt.mask).abs().sum().item() == 0):
                raise RuntimeError("Missing, non-finite, or zero border gradient")
            optimizer.step()
            total_steps += 1
            updates_in_epoch += 1
            samples_seen += labels.numel()
            loss_sum += float(loss.detach()) * labels.numel()
            if total_steps % 20 == 0:
                print(f"Training update {total_steps}: loss={loss.item():.4f}", flush=True)
        torch.cuda.synchronize()
        epoch_train_seconds = time.perf_counter() - tick
        train_seconds += epoch_train_seconds
        if updates_in_epoch == 0:
            break

        torch.cuda.synchronize()
        tick = time.perf_counter()
        accuracy = evaluate_validation(model, prompt, validation_loader, text_features, device)
        torch.cuda.synchronize()
        epoch_val_seconds = time.perf_counter() - tick
        val_seconds += epoch_val_seconds
        mean_loss = loss_sum / samples_seen
        history.append(dict(epoch=epoch, updates=total_steps,
                            val_accuracy=accuracy, mean_train_loss=mean_loss,
                            train_seconds=epoch_train_seconds,
                            validation_seconds=epoch_val_seconds))
        print(f"Epoch {epoch}/{args.epochs}: mean_train_loss={mean_loss:.4f}, "
              f"full_val_accuracy={accuracy:.3f}%, updates={total_steps}", flush=True)
        if accuracy > best_score:  # Earliest epoch wins a tie.
            best_score = accuracy
            best_epoch = epoch
            best_state = cpu_checkpoint(prompt)
            print(f"  New best checkpoint: epoch {best_epoch} ({best_score:.3f}%)", flush=True)
        if args.max_steps and total_steps >= args.max_steps:
            break

    run_dir.mkdir(parents=True, exist_ok=False)
    torch.save({"state_dict": best_state, "config": scientific_config,
                "best_epoch": best_epoch, "best_val_accuracy": best_score},
               run_dir / "best_prompt.pt")
    restored = torch.load(run_dir / "best_prompt.pt", map_location="cpu", weights_only=True)
    prompt.load_state_dict(restored["state_dict"], strict=True)
    torch.cuda.synchronize()
    tick = time.perf_counter()
    restored_val = evaluate_validation(model, prompt, validation_loader, text_features, device)
    torch.cuda.synchronize()
    val_seconds += time.perf_counter() - tick
    if not math.isclose(restored_val, best_score, abs_tol=1e-5):
        raise RuntimeError(f"Best checkpoint reload mismatch: {restored_val} != {best_score}")
    if best_epoch > 0 and torch.count_nonzero(best_state["raw_delta"]).item() == 0:
        raise RuntimeError("Selected trained checkpoint has no learned parameters")

    output = {
        "status": "development_validation_only",
        "config": scientific_config,
        "initial_val_accuracy": initial_val,
        "best_val_accuracy": best_score,
        "best_epoch": best_epoch,
        "reloaded_best_val_accuracy": restored_val,
        "optimizer_steps": total_steps,
        "history": history,
        "timing_seconds": {
            "train_loops": train_seconds,
            "validation_including_recheck": val_seconds,
            "total_including_load": time.perf_counter() - start_total,
        },
        "effective_trainable_parameters": prompt.effective_trainable_parameters(),
        "python_torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "hardware": torch.cuda.get_device_name(0),
        "test_partition_evaluated": False,
        "warning": "Validation-only development run; do NOT include in the 93-run baseline results.",
    }
    (run_dir / "training.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Selected best epoch {best_epoch}: {best_score:.3f}% (reloaded: {restored_val:.3f}%)", flush=True)
    print(f"Success: BEST validation checkpoint saved under {run_dir}; TEST NOT USED", flush=True)


if __name__ == "__main__":
    main()
