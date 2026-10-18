# P25: Few-Shot CLIP Adaptation

**CISC 473 Deep Learning Capstone - Fall 2026 - Group 17**

Compare efficient CLIP adaptation methods on **EuroSAT**, **DTD**, and
**Oxford Flowers102**, and investigate whether multi-resolution training of a
shared pixel-space visual prompt improves resolution robustness on EuroSAT.

The project examines accuracy, sensitivity to few-shot sampling and prompt design,
and adaptation cost under limited labeled data. The visual-prompt extension is a
research hypothesis that still needs implementation, evaluation and team review.

## Team

| Member | Role | Contribution record |
| --- | --- | --- |
| Miller Gao | Project Lead | **TODO:** add merged contributions, analysis/report work and current integration status. |
| Wangyi Lu | Research Lead | Baseline and CoOp sensitivity experiments completed locally; final contribution statement to be reviewed with the team. |
| Yubo Wu | Engineering Lead | **TODO:** add contributed modules, data/evaluation work and visual-prompt implementation status after integration. |

**TODO:** confirm remaining method owners and final contribution percentages.
These role labels preserve the original team introduction; contribution entries
should describe work actually completed and reviewed.

## Current status and methods

The main **93-run baseline matrix** and **12-run CoOp sensitivity study** have
completed locally. This code submission contains implementations, configurations,
tests and reproducibility instructions. Result tables, figures, generated
manifests and report materials will be reviewed and submitted separately.

| Method | Adaptation | Current repository status |
| --- | --- | --- |
| Zero-shot CLIP (ViT-B/32) | Fixed textual class descriptions | Implemented; evaluated locally on all three datasets |
| CoOp | Learn shared text-context vectors with CLIP frozen | Simplified baseline implemented; main and sensitivity runs completed locally |
| Original Tip-Adapter | Fixed labeled feature cache, without gradient training | Single-view baseline implemented; main runs completed locally; not Tip-Adapter-F |
| Supervised ResNet-50 | Fine-tune an ImageNet-pretrained model on matched few-shot samples | **TODO:** confirm owner and integrate implementation, configs and results |
| Standard pixel-space visual prompting | Learn a shared input-image modification | **TODO:** integrate and validate the existing-method baseline |
| Multi-resolution pixel-space visual prompting | Train the shared prompt across effective resolutions | **TODO:** implement and evaluate the proposed extension |

These are simplified project baselines, not strict reproductions of published
scores. CoOp uses deterministic cached image features and omits upstream
augmentation/warmup. Tip-Adapter uses a single-view fixed cache. Algorithm sources
and implementation differences are recorded in [UPSTREAM.md](docs/UPSTREAM.md).

## Shared experimental protocol

Read [PROTOCOL.md](docs/PROTOCOL.md) before integrating or comparing methods.

- Use **1, 2, 4, 8 and 16 training images per class** and sampling/training seeds
  **1, 2 and 3** for adapted methods. Zero-shot is evaluated once per dataset.
- Share the CLIP checkpoint, class ordering, disjoint train/validation/test
  partitions and sampled support lists across applicable methods.
- Count validation labels separately from the k-shot training budget. Select
  hyperparameters and checkpoints using validation data only.
- Report top-1 accuracy as mean and sample standard deviation across seeds,
  trainable parameters, adaptation time, cache costs where applicable and hardware.
- Keep frozen-feature extraction separate from pixel-prompt training. Training a
  pixel prompt requires gradients through the frozen image encoder; cached image
  embeddings cannot train an image-level prompt.

| Dataset | Partition protocol | Validation supervision |
| --- | --- | --- |
| EuroSAT RGB | Teammate's unchanged script; 60/10/30, split seed 2026; nested support lists for seeds 1/2/3 | All 2,700 validation labels |
| DTD | Official partition 1 | Four validation images/class, 188 total |
| Flowers102 | Custom stratified 50/20/30 over all images, split seed 0 | Four validation images/class, 408 total |

The EuroSAT importer checks the dataset ordering fingerprint. Generate its shared
lists before running experiments. Flowers102's official training split has only
10 images/class; the custom split supports 16-shot and differs from both that
official split and the exact published Zhou split.

**TODO:** confirm these protocols before merging teammate results. Document
any agreed changes, method-specific preprocessing and experiments that must be rerun.

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
Use `run-check.ps1 --device cpu` when checking a CPU environment.

On Linux/Colab, install matching torch/torchvision wheels for the device, then
install `setuptools<81`, `wheel` and `requirements.txt`. Replace `run.ps1` with
`python`. Keep dataset downloads and outputs on persistent storage.

`requirements.experiment.lock.txt` records the Windows experiment environment,
including SciPy 1.18.1 for Flowers102. Its CUDA wheel versions require the matching
PyTorch index; it is not a universal environment lock. Optional plotting
dependencies are listed separately in `requirements-report.txt`.

On the existing experiment computer, ignored local settings may point to an
already-tested Python environment and CLIP cache. New clones should create their
own environment with setup; personal paths are not part of this submission.

**TODO:** add and validate any additional dependencies introduced by
teammate modules, and record the supported hardware/installation commands.

## Reproduce available experiments

Run from the project root:

```powershell
.\run.ps1 scripts/sweep.py --config configs/baselines.json --download --dry-run
.\run.ps1 scripts/sweep.py --config configs/baselines.json --download
.\run.ps1 scripts/summarize.py
.\run.ps1 scripts/sweep.py --config configs/coop_sensitivity.json --output-root results/sensitivity-raw
.\run.ps1 scripts/summarize.py --root results/sensitivity-raw --output results/coop-sensitivity.csv
```

The main matrix has three zero-shot evaluations and 90 adapted runs: three
datasets, two adapted methods, five budgets and three seeds. Sensitivity has
12 EuroSAT 16-shot runs comparing random context lengths 4/8/16 and random versus
word initialization at length 4. Matching completed configurations are skipped;
resume with the same command without `--overwrite`.

**TODO:** add runnable ResNet-50 and pixel-prompt commands and their
config paths after the implementations are merged and checked. Keep the current
baseline commands reproducible.

## Team integration: remaining work

- [ ] **ResNet-50:** confirm the owner; add the ImageNet weight version, classifier
  replacement, normalization, fine-tuning schedule and matched-support configs.
- [ ] **Standard pixel prompting:** specify prompt structure/size, application
  before normalization, constraints, optimizer and number of training updates.
- [ ] **Multi-resolution extension:** train with effective resolutions 16/32/64;
  evaluate native inputs, 16/32/64 and unseen resolution 24. Match support lists,
  prompt structure, optimizer and update budgets with the standard condition.
- [ ] **Evaluation integration:** align metric schemas, class ordering, seed
  handling, validation selection and adaptation-time accounting across methods.
- [ ] **Compatibility checks:** add small end-to-end checks for new modules and
  verify whether any existing experiments are affected by implementation changes.

**TODO(team):** replace these checklist items with reviewed implementation details
and commands as work is integrated. Leave unfinished items clearly marked.

## Results, analysis and final deliverables

The baseline and sensitivity summaries exist locally. The current code-only
ignore rules exclude results, figures, generated manifests and report files.

**TODO:** add the following after the reviewed result submission:

- Accuracy-versus-shots tables/figures with seed counts and uncertainty.
- Prompt sensitivity and efficiency comparisons with the timing definition.
- ResNet-50 and visual-prompt results under the shared protocol.
- Resolution robustness, including the unseen-resolution evaluation.
- Confusion matrices, inspected image/retrieval examples and supported failure
  analysis; distinguish observations from hypotheses about their causes.
- Conclusions, limitations, final report/presentation links and verified
  contribution percentages.

Link result artifacts only after they are committed. Preserve valid baseline
results; rerun experiments affected by corrected implementations or changed
scientific settings. The proposed extension's benefit remains unverified until
its experiments are complete.

## Code layout and verification

- `src/make_eurosat_splits.py`: teammate's original, unchanged split generator.
- `src/p25/data.py`: dataset loading, manifests, overlap/fingerprint checks and download retries.
- `src/p25/methods.py`: shared-context CoOp and original training-free Tip-Adapter.
- `scripts/`: individual experiments, sweeps, aggregation and optional reporting.
- `configs/`: pilot, baseline matrix and CoOp sensitivity settings.
- `tests/`: data protocols, method equations, prompt gradients, statistics and download recovery.

```powershell
.\run.ps1 -m unittest discover -s tests -v
```

Real CLIP tests require `P25_TEST_CLIP_CACHE` pointing to a directory containing
`ViT-B-32.pt`. EuroSAT integration tests require `P25_TEST_DATA` pointing to the
prepared data root. These tests explicitly skip when their resources are absent.

After both summaries and raw runs exist locally, `scripts/make_report.py` can audit
them and generate scientific PNG/SVG figures and a Markdown report. Install the
optional `requirements-report.txt` first; this command does not train models.

**TODO:** document extra tests, required resources and known
limitations introduced by new implementations.

## Git and submission scope

Use this project folder for development, experiments and Git. Local `main` tracks
`origin/main`; see [GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md) for submission commands.

The current `.gitignore` excludes datasets, environments, caches, weights, local
settings, all results/manifests, historical/course documents and generated reports.
Only the protocol, upstream attribution and Git workflow are included from `docs/`.
Review and deliberately update the relevant ignore rules when submitting
scientific evidence later; the local files remain available.

## References

1. Radford et al. (2021), [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020).
2. Zhou et al. (2022), [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134).
3. Zhang et al. (2022), [Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification](https://arxiv.org/abs/2207.09519).
4. Jia et al. (2022), [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119).
5. Bahng et al. (2022), [Exploring Visual Prompts for Adapting Large-Scale Models](https://arxiv.org/abs/2203.17274).
6. Corley et al. (2023), [Revisiting pre-trained remote sensing model benchmarks: resizing and normalization matters](https://arxiv.org/abs/2305.13456).

**TODO:** add sources used by newly integrated
methods, verify citation metadata and retain upstream attribution/license notices.

## AI usage and contribution declaration

ChatGPT assisted background explanations, proposal drafting, repository
organization, code generation, tests, local pilot execution, and documentation. The user ran the full baseline and sensitivity sweeps.
Team members must review and understand the code and interpret the actual results.

**TODO:** add the AI assistance actually used in their contribution.
**TODO:** review this disclosure and the final contribution declaration
against the course requirements before submitting the report.
