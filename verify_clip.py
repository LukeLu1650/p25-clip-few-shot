"""Check GPU computation and official CLIP inference; not an accuracy benchmark."""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import clip
import torch
import torchvision
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, help="Optional real image for inference")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--clip-cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    clip_cache = args.clip_cache or root / ".cache" / "clip"
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the installed PyTorch build and driver.")

    print(f"Python: {platform.python_version()}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"Torchvision: {torchvision.__version__}", flush=True)
    print(f"PyTorch CUDA runtime: {torch.version.cuda}", flush=True)
    print(f"Device: {device}", flush=True)
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else None
    print(f"GPU: {gpu}", flush=True)

    # Exercise an actual GPU kernel and backward pass, not just GPU detection.
    matrix = torch.randn(64, 64, device=device, requires_grad=True)
    value = (matrix @ matrix.T).square().mean()
    value.backward()
    assert matrix.grad is not None and torch.isfinite(matrix.grad).all()
    print("Tensor forward/backward: PASS", flush=True)

    model, preprocess = clip.load(
        "ViT-B/32", device=device, jit=False, download_root=str(clip_cache)
    )
    model.eval()
    image = (
        Image.open(args.image).convert("RGB")
        if args.image
        else Image.new("RGB", (224, 224), color=(128, 128, 128))
    )
    images = preprocess(image).unsqueeze(0).to(device)
    texts = clip.tokenize([
        "a satellite image of a forest",
        "a satellite image of a river",
        "a satellite image of a residential area",
    ]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        logits, _ = model(images, texts)
        probabilities = logits.float().softmax(dim=-1)
    assert tuple(image_features.shape) == (1, 512)
    assert tuple(text_features.shape) == (3, 512)
    assert tuple(logits.shape) == (1, 3)
    assert all(torch.isfinite(x).all() for x in [image_features, text_features, logits, probabilities])
    assert torch.allclose(probabilities.sum(-1), torch.ones(1, device=device), atol=1e-5)
    print(f"Image feature shape: {tuple(image_features.shape)}", flush=True)
    print(f"Text feature shape: {tuple(text_features.shape)}", flush=True)
    print(f"Logit shape: {tuple(logits.shape)}", flush=True)
    print("CLIP image/text inference: PASS", flush=True)

    weights = clip_cache / "ViT-B-32.pt"
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": device,
        "gpu": gpu,
        "model": "OpenAI CLIP ViT-B/32",
        "clip_commit": "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6",
        "clip_module": clip.__file__,
        "weights_sha256": digest.hexdigest(),
        "synthetic_image": args.image is None,
        "image_feature_shape": list(image_features.shape),
        "text_feature_shape": list(text_features.shape),
        "logit_shape": list(logits.shape),
        "tensor_forward_backward": "PASS",
        "clip_inference": "PASS",
        "note": "An environment check only; no EuroSAT accuracy or training result is claimed.",
    }
    output = args.output or root / "environment-check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {output}", flush=True)
    print("ALL CHECKS PASSED. This is not a dataset accuracy evaluation.", flush=True)


if __name__ == "__main__":
    main()
