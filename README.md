# P25: Few-Shot CLIP Adaptation

CISC 473, Fall 2026, Group 17. Compare zero-shot OpenAI CLIP ViT-B/32,
shared-context CoOp and original training-free Tip-Adapter on EuroSAT RGB,
DTD and Oxford Flowers102. The planned extension studies multi-resolution
pixel-space visual prompting on EuroSAT.

**Status:** baseline code, configurations and tests are ready. The main 93-run
baseline matrix and 12-run CoOp sensitivity study have completed locally.
This submission contains code and reproducibility instructions; result tables,
figures, generated manifests and report materials will be submitted separately.
ResNet-50, pixel prompting and resolution-robustness experiments remain team work.

Use this folder for development, experiments and Git. Local `main` tracks
`origin/main`; see [Git workflow](docs/GIT_WORKFLOW.md).

## Setup

Use Python 3.12. On Windows:

```powershell
.\setup.ps1
.\run-check.ps1
.\run.ps1 src/make_eurosat_splits.py
.\run.ps1 scripts/sweep.py --config configs/pilot.json
```

Setup creates a local `.venv` and installs torch 2.7.1 / torchvision 0.22.1
with CUDA 12.8 wheels. For CPU, run `setup.ps1 -TorchBuild cpu` and add
`--device cpu` to experiment commands. If needed, pass
`setup.ps1 -BootstrapPython <path-to-python-3.12>`.

On Linux/Colab, install matching torch/torchvision wheels for the device, then
install `setuptools<81`, `wheel` and `requirements.txt`. Replace `run.ps1` with
`python`. Keep dataset downloads and outputs on persistent storage.

`requirements.experiment.lock.txt` records the Windows experiment environment,
including SciPy 1.18.1 for Flowers102. Its CUDA wheel versions require the matching
PyTorch index; it is not a universal environment lock. Optional plotting
dependencies are listed separately in `requirements-report.txt`.

On the existing experiment computer, ignored local settings may point to an
already-tested Python environment and CLIP cache. New clones must create their
own environment with setup; personal paths are not part of this submission.

## Reproduce experiments

Read [PROTOCOL.md](docs/PROTOCOL.md) before running comparisons.

```powershell
.\run.ps1 scripts/sweep.py --config configs/baselines.json --download --dry-run
.\run.ps1 scripts/sweep.py --config configs/baselines.json --download
.\run.ps1 scripts/summarize.py
.\run.ps1 scripts/sweep.py --config configs/coop_sensitivity.json --output-root results/sensitivity-raw
.\run.ps1 scripts/summarize.py --root results/sensitivity-raw --output results/coop-sensitivity.csv
```

The main matrix has three zero-shot evaluations and 90 adapted runs: three
datasets, two adapted methods, five shot budgets and three seeds. Sensitivity
has 12 EuroSAT 16-shot runs. Matching completed configurations are skipped.
Resume with the same command without `--overwrite`.

Shared data rules:

- EuroSAT uses the teammate's unchanged split script: 60/10/30, split seed 2026,
  nested support lists for seeds 1/2/3 and all 2,700 validation labels. Run the
  split generator first; its output ordering fingerprint is checked by the importer.
- DTD uses official partition 1 with four validation examples/class.
- Flowers102 uses a custom stratified 50/20/30 split, seed 0, and four validation
  examples/class. This supports 16-shot; it is not the official 10-training-images/
  class split or the exact published Zhou split.
- Methods must share class order, image partitions and support lists. Validation
  labels are counted separately. Select hyperparameters and checkpoints on
  validation data only.

These are simplified project baselines, not strict reproductions of published
scores. CoOp uses cached deterministic image features and omits upstream
augmentation/warmup. Tip-Adapter uses a single-view fixed cache and is not
Tip-Adapter-F. See [UPSTREAM.md](docs/UPSTREAM.md) for sources and differences.
Pixel-prompt training requires images and gradients through the frozen image
encoder; cached embeddings cannot train an image-level prompt.

## Code and verification

- `src/make_eurosat_splits.py`: teammate's original, unchanged split generator.
- `src/p25/data.py`: dataset loading, manifests, overlap/fingerprint checks and download retries.
- `src/p25/methods.py`: shared-context CoOp and original training-free Tip-Adapter.
- `scripts/`: individual experiments, sweeps, summary aggregation and optional reporting.
- `configs/`: pilot, baseline matrix and CoOp sensitivity settings.
- `tests/`: data protocols, method equations, prompt gradients, statistics and download recovery.

```powershell
.\run.ps1 -m unittest discover -s tests -v
```

Real CLIP tests require `P25_TEST_CLIP_CACHE` pointing to a directory containing
`ViT-B-32.pt`. EuroSAT integration tests require `P25_TEST_DATA` pointing to the
prepared data root. These tests explicitly skip when their resources are absent.

After both summaries and raw runs exist locally, `scripts/make_report.py` can audit
them and produce scientific PNG/SVG figures and a Markdown report. Install the
optional `requirements-report.txt` first; this command does not train models.

## Submission scope and team

The current `.gitignore` excludes datasets, environments, caches, weights, local
settings, all results/manifests, historical/course documents and generated reports.
Only the protocol, upstream attribution and Git workflow are included from `docs/`.
Review and deliberately update the ignore rules when submitting scientific
evidence later; the local files remain available.

Miller Gao is Project Lead, Wangyi Lu is Research Lead, and Yubo Wu is Engineering
Lead. Confirm method ownership and integration responsibilities together against
the submitted proposal.

ChatGPT assisted planning, code generation, tests, local pilot execution,
documentation and descriptive analysis. The user ran the full baseline and
sensitivity sweeps. Team members must review and understand the implementation,
interpret actual results, and disclose accurate contributions and AI use.
