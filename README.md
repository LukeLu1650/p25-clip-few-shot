# P25: CLIP-Based Efficient Few-Shot Transfer to Specialised Image Domains

**CISC 473 Deep Learning Capstone · Fall 2026 · Group 17**

This project studies efficient few-shot adaptation of **CLIP ViT-B/32** on
**EuroSAT**, **DTD**, and **Oxford Flowers102**. Following proposal feedback,
our main research focus is **resolution robustness of pixel-space visual
prompting on EuroSAT**; the three-dataset CLIP baselines provide supporting
comparisons.

**Research question:** Can one shared pixel-space visual prompt trained across
multiple effective image resolutions improve robustness to resolution
degradation while preserving accuracy at the native resolution?

The midterm results show a trade-off: multi-resolution training improves mean
validation accuracy across resolutions, but reduces native-resolution accuracy
relative to Single-64. Held-out test evaluation remains the next major step.

## Current progress

The shared-protocol baseline experiments, CoOp sensitivity study, and initial
three-seed visual-prompt comparison are complete. The results and contribution
statements below follow the Group 17 midterm report. The repository also includes
experiment configurations, protocol checks, aggregation tools, and prompt
training scripts.

| Component | Status |
| --- | --- |
| Zero-shot CLIP | Completed on all three datasets |
| CoOp | Completed at 1, 2, 4, 8, and 16 shots/class with three seeds per setting |
| Original Tip-Adapter | Completed under the same datasets, budgets, and seeds |
| CoOp sensitivity | Completed: 12 EuroSAT 16-shot runs covering context length and initialization |
| Results and analysis | Local summaries, accuracy/efficiency figures, confusion matrices, and prediction records prepared |
| Single-resolution pixel prompting | Completed Single-16, Single-32, and Single-64 validation experiments, each with three seeds |
| Multi-resolution pixel prompting | Completed one shared 16/32/64 prompt condition with three seeds; validation results reported below |
| Final prompt robustness evaluation | Remaining: held-out test evaluation, including unseen resolution 24×24 |
| Supervised ResNet-50 | Remaining: implementation and comparison are not included in the current repository |

The main matrix contains **93 completed runs**: three zero-shot evaluations and
90 few-shot runs (three datasets × two adaptation methods × five budgets × three
seeds). The sensitivity study contains **12 additional executions**; its
16-token random-initialization condition repeats the main CoOp setting. A
separate set of **12 visual-prompt runs** compares four training strategies
across three seeds at the EuroSAT 16-shot setting.

## Team

| Member | Role | Midterm contributions | Share |
| --- | --- | --- | ---: |
| Miller Gao | Project Lead | Literature review, proposal and experiment planning; single-/multi-resolution pixel-prompt implementation and preliminary validation experiments; visual-prompt results preparation | 33% |
| Wangyi Lu | Research Lead | Experimental planning; zero-shot CLIP, CoOp, and Tip-Adapter implementation; baseline and CoOp sensitivity experiments; methodology writing | 33% |
| Yubo Wu | Engineering Lead | Data preparation, pipeline and shared-split coordination; collaboration on the visual-prompt pipeline; evaluation and selected-baseline reproduction work; qualitative analysis and preliminary-results writing | 34% |

## Research context

Our literature review covers nine works, matching the midterm report:

| Work | Connection to this project |
| --- | --- |
| CLIP [1] | Frozen image/text backbone and zero-shot reference |
| CoOp [2] | Learnable text-context baseline |
| Tip-Adapter [3] | Training-free labeled-feature-cache baseline |
| Visual Prompt Tuning [4] | Visual-token adaptation, distinct from our input-pixel prompt |
| Pixel-space visual prompting [5] | Foundation for adapting frozen models through shared image modifications |
| Corley et al. [6] | Motivation for controlling resizing and normalization in remote sensing |
| GDA [7] | Training-free adaptation using class statistics; source of published ViT-B/32 reference values |
| Enhanced Visual Prompting (EVP) [8] | Related work on pixel-prompt placement and optimization |
| LR0.FM / LR-TK0 [9] | Low-resolution robustness and token-based adaptation; related objective with different learnable components |

Our contribution is a controlled empirical comparison of single- versus
multi-resolution **pixel-space border prompts** under matched few-shot samples
and training budgets. We do not claim that multi-resolution training itself is
new. GDA, EVP, and LR0.FM are related-work references rather than additional
implemented baselines.

## Baseline results

Top-1 accuracy (%) from the completed local experiments:

| Dataset | Zero-shot CLIP | CoOp, 16-shot | Tip-Adapter, 16-shot |
| --- | ---: | ---: | ---: |
| EuroSAT | 46.77 | 82.76 ± 0.93 | 72.21 ± 1.34 |
| DTD | 42.45 | 65.57 ± 0.41 | 58.30 ± 1.20 |
| Flowers102 | 66.26 | 92.29 ± 0.15 | 88.62 ± 0.76 |

Adapted results are mean ± sample standard deviation over seeds **1, 2, and 3**.
Zero-shot is evaluated once per dataset. CoOp has the highest observed mean at
16-shot on all three datasets; the relative performance varies at lower budgets.

These are **project baseline results** under the protocols below. CoOp uses
deterministic cached image features and omits the full upstream augmentation and
warmup recipe. Tip-Adapter uses a single-view fixed cache and does not include
Tip-Adapter-F. See [upstream implementation notes](docs/UPSTREAM.md).

### Published reference comparison

The midterm report compares our results with the ViT-B/32 entries in Table 9 of
[Wang et al. (2024)](https://arxiv.org/html/2402.04087v1).

| Method | Source | EuroSAT | DTD | Flowers102 |
| --- | --- | ---: | ---: | ---: |
| Zero-shot CLIP | Published | 45.19 | 43.97 | 66.95 |
| Zero-shot CLIP | Ours | 46.77 | 42.45 | 66.26 |
| CoOp, 16-shot | Published | 83.43 | 65.37 | 94.97 |
| CoOp, 16-shot | Ours | 82.76 ± 0.93 | 65.57 ± 0.41 | 92.29 ± 0.15 |
| Tip-Adapter, 16-shot | Published | 69.53 | 61.90 | 91.61 |
| Tip-Adapter, 16-shot | Ours | 72.21 ± 1.34 | 58.30 ± 1.20 | 88.62 ± 0.76 |

CoOp differs from these reference means by 0.67 percentage points on EuroSAT
and 0.20 on DTD. Some partitions and adaptation settings differ, so numerical
agreement is contextual evidence rather than an exact reproduction of the
published protocol. Selected published-protocol reproduction remains a separate
work item.

### CoOp sensitivity on EuroSAT

All conditions use 16 training images/class and three seeds.

| Context tokens | Initialization | Accuracy (%) |
| --- | --- | ---: |
| 4 | Random | 82.51 ± 1.02 |
| 8 | Random | 83.26 ± 1.78 |
| 16 | Random | 82.76 ± 0.93 |
| 4 | “a photo of a” | 82.85 ± 0.52 |

The highest observed mean occurs at eight context tokens, while four-token word
initialization has the lowest observed standard deviation. These three-seed
comparisons are descriptive and do not establish statistical significance.

## Experimental protocol

Full details are in [PROTOCOL.md](docs/PROTOCOL.md).

- **Backbone:** official OpenAI CLIP ViT-B/32; CLIP parameters remain frozen.
- **Few-shot budgets:** 1, 2, 4, 8, and 16 labeled training images/class.
- **Repeated experiments:** sampling/training seeds 1, 2, and 3.
- **Comparability:** applicable methods share class ordering, fixed partitions,
  and the exact training sample lists for each seed and budget.
- **Model selection:** validation data only; validation labels are counted
  separately from the k-shot training budget.
- **Reporting:** top-1 accuracy, sample standard deviation, trainable parameters,
  and adaptation time, including feature extraction/cache construction where
  applicable.

| Dataset | Partition | Training pool | Validation pool | Validation used | Test |
| --- | --- | ---: | ---: | ---: | ---: |
| EuroSAT RGB | Shared 60/10/30, split seed 2026 | 16,200 | 2,700 | 2,700 | 8,100 |
| DTD | Official partition 1 | 1,880 | 1,880 | 188 | 1,880 |
| Flowers102 | Custom approximately 50/20/30, split seed 0 | 4,093 | 1,633 | 408 | 2,463 |

Few-shot support images are sampled from the training pools; pool sizes are not
the number of images used to train each few-shot model. DTD and Flowers102 use
four validation images/class. Support sets are class-balanced and nested across
shot counts within each seed. Pixel prompting uses the same EuroSAT 16-shot
support lists as the corresponding baselines.

The Flowers102 custom split supports 16-shot experiments and differs from both
the official training split and the exact published Zhou split. DTD uses its
official partition rather than CoOp's random partition. These differences matter
when comparing against published scores.

CoOp uses cross-entropy, SGD with momentum 0.9, learning rate 0.002, cosine
annealing without warmup, batch size 32, and 50 epochs. The main setting learns
16 shared context tokens with normal initialization of standard deviation 0.02.
The earliest best-validation checkpoint is retained. Tip-Adapter selects
`alpha` from `{0, 0.1, 1, 2, 4, 8}` and `beta` from `{0.5, 1, 2, 4, 8, 16}`
using validation accuracy only.

Baseline timing uses `cold_adaptation_seconds`: original feature-extraction
durations plus text encoding and adaptation/validation selection. Model
download/load and test inference are excluded. Cached extraction durations are
reused for accounting; the timings are not independent cold-start replications.

## Setup

Use **Python 3.12**. Clone the repository and run commands from its root:

```powershell
git clone https://github.com/LukeLu1650/p25-clip-few-shot.git
cd p25-clip-few-shot
.\setup.ps1
.\run-check.ps1
```

Windows setup creates `.venv` and installs PyTorch 2.7.1 / torchvision 0.22.1
with CUDA 12.8. If Python is not found, supply its path with
`setup.ps1 -BootstrapPython <path-to-python-3.12>`.

For CPU baseline experiments, use `setup.ps1 -TorchBuild cpu`,
`run-check.ps1 --device cpu`, and `--device cpu` on experiment commands.
The pixel-prompt training scripts require CUDA.

On Linux, install compatible PyTorch/torchvision wheels, then `setuptools<81`,
`wheel`, and `requirements.txt`. Use `python` in place of `.\run.ps1`.
`requirements.experiment.lock.txt` records the Windows experiment environment;
its CUDA packages require the matching PyTorch package index.

## Run baseline and sensitivity experiments

Prepare the shared EuroSAT lists, then optionally run the pilot:

```powershell
.\run.ps1 src/make_eurosat_splits.py
.\run.ps1 scripts/sweep.py --config configs/pilot.json --download
```

Run the main matrix and aggregate its results:

```powershell
.\run.ps1 scripts/sweep.py --config configs/baselines.json --download --dry-run
.\run.ps1 scripts/sweep.py --config configs/baselines.json --download
.\run.ps1 scripts/summarize.py
```

Run and summarize the CoOp sensitivity study:

```powershell
.\run.ps1 scripts/sweep.py --config configs/coop_sensitivity.json --output-root results/sensitivity-raw
.\run.ps1 scripts/summarize.py --root results/sensitivity-raw --output results/coop-sensitivity.csv
```

Matching completed baseline/sensitivity configurations are skipped. Resume with
the same command; use `--overwrite` only when deliberately replacing a run.

## Pixel-space visual prompting

`src/p25/pixel_prompt.py` implements a shared additive RGB border prompt. Its
default border width is eight pixels, with a tanh-bounded perturbation applied
in RGB space, clipped to [0, 1], and converted back to CLIP-normalized input.
Both CLIP encoders and the text template stay fixed; gradients pass through the
image encoder to update the prompt.

The perturbation is `0.25 * tanh(theta) * border_mask`, with zero initialization.
There are **20,736 effective trainable RGB border parameters**, stored in a
150,528-element tensor whose interior is masked out. The text classifier uses
the fixed template `a satellite image of {class}.` and the pretrained CLIP logit
scale.

The comparison runner uses the same support lists, optimizer, batch size, and
update budget for both conditions. The input is resized to an effective
resolution before standard CLIP preprocessing to 224×224:

- **Single-16 / Single-32 / Single-64:** three separate prompts, each trained at
  its fixed effective resolution.
- **Multi-resolution:** one shared prompt; each training image independently samples 16×16,
  32×32, or 64×64 with equal probability.

### Completed training settings

| Setting | Value |
| --- | --- |
| Dataset and support | EuroSAT; 16 images/class, 160 images total |
| Seeds | 1, 2, 3 |
| Optimizer | AdamW, learning rate 0.01, weight decay 0 |
| Schedule | Constant learning rate |
| Training / validation batch size | 2 / 8 |
| Training budget | 5 epochs; 400 updates per run |
| Loss / precision | Cross-entropy / FP32 |
| Selection data | Full 2,700-image validation partition |

All four strategies select checkpoints by the same score: mean validation
accuracy at effective resolutions **16, 32, and 64**, computed within each seed.
Epoch 0 is included, ties select the earliest epoch, and saved checkpoints are
reloaded to verify their selection scores. No random flips, crops, or color
jitter are used; multi-resolution training adds only the specified resolution
sampling.

### Preliminary validation results

The following reproduces Table 5 of the midterm report. Values are top-1
accuracy (%) as mean ± sample standard deviation over three seeds.

| Training strategy | Evaluate at 16×16 | Evaluate at 32×32 | Evaluate at 64×64 |
| --- | ---: | ---: | ---: |
| Single-16 | **48.98 ± 10.03** | 50.56 ± 4.98 | 54.52 ± 4.12 |
| Single-32 | 35.27 ± 2.77 | **63.20 ± 2.73** | 63.07 ± 1.45 |
| Single-64 | 22.89 ± 3.79 | 48.41 ± 1.43 | **70.90 ± 2.21** |
| Multi-resolution | 46.11 ± 2.12 | 60.31 ± 1.98 | 61.85 ± 0.84 |

Multi-resolution training has the highest cross-resolution mean validation
accuracy: **56.09 ± 1.12%**, compared with **53.85 ± 1.93%** for Single-32,
the strongest single-resolution condition on this composite metric. Each
seed's cross-resolution mean is calculated before aggregation across seeds.

Relative to Single-64, multi-resolution training gains **23.22 percentage
points at 16×16** and loses **9.05 percentage points at 64×64**. The
64-to-16 accuracy drop decreases from 48.01 to 15.74 percentage points; this
smaller drop partly reflects lower native-resolution accuracy. A prompt trained
at each individual resolution still has the highest mean at that resolution.

These results support a preliminary robustness–accuracy trade-off. They use the
same validation images as checkpoint selection, so they are not held-out test
evidence. Test images and **unseen resolution 24×24** remain excluded from
training and selection; test evaluation at native resolution and effective
resolutions 16, 24, 32, and 64 is planned after configurations are frozen.

### Run the controlled comparison

Example matched 16-shot, seed-1 development runs:

```powershell
.\run.ps1 scripts/train_resolution_prompts.py --mode single --train-resolution 16 --shots 16 --seed 1 --epochs 5
.\run.ps1 scripts/train_resolution_prompts.py --mode single --train-resolution 32 --shots 16 --seed 1 --epochs 5
.\run.ps1 scripts/train_resolution_prompts.py --mode single --train-resolution 64 --shots 16 --seed 1 --epochs 5
.\run.ps1 scripts/train_resolution_prompts.py --mode multi --shots 16 --seed 1 --epochs 5
```

Repeat with seeds 2 and 3 for the full 12-run comparison. These scripts save
development results under `results/resolution_prompt_dev/` and refuse to
overwrite an existing run.

`scripts/train_pixel_prompt.py` also provides single-resolution development
training with checkpoint selection at one validation resolution. Use the
comparison runner above for the matched single-/multi-resolution comparison.

Keep validation-only prompt results separate from baseline test summaries.

## Qualitative findings

The midterm report includes row-normalized CoOp and Tip-Adapter confusion
matrices at 16-shot, pooled over three seeds on the same **8,100 EuroSAT test
images**. Overall accuracy does not imply uniform superiority across classes:

- Herbaceous vegetation accuracy is 81.1% for CoOp and 63.6% for Tip-Adapter;
  confusion with forest is 6.4% and 22.5%, respectively.
- Industrial accuracy is 86.3% for CoOp and 95.4% for Tip-Adapter.
- Selected prediction examples illustrate CoOp-only correct, Tip-Adapter-only
  correct, and shared-error cases. They were deliberately selected and cannot
  be used to estimate the overall error rate.

## Analysis and verification

Local outputs include the baseline and sensitivity CSV summaries, prediction and
retrieval records, confusion matrices, accuracy/efficiency plots, and analysis
checks. To rebuild the baseline analysis after the summaries and raw runs exist:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-report.txt
.\run.ps1 scripts/make_report.py
```

The report generator checks the expected run counts, recomputes summary
statistics, and generates figures and a Markdown results report. It does not
train models or summarize the development pixel-prompt runs.

Run the automated checks with:

```powershell
.\run.ps1 -m unittest discover -s tests -v
```

Real-CLIP checks require `P25_TEST_CLIP_CACHE` pointing to the directory containing
`ViT-B-32.pt`; EuroSAT integration checks require `P25_TEST_DATA` pointing to the
prepared data root. Resource-dependent checks skip when those resources are
unavailable.

## Repository layout

```text
configs/                 Pilot, baseline matrix, and sensitivity configurations
docs/                    Protocol, upstream attribution, and Git workflow
scripts/                 Experiments, aggregation, reporting, and prompt training
src/make_eurosat_splits.py  Shared EuroSAT split generator
src/p25/data.py          Data loading, shared manifests, and dataset checks
src/p25/methods.py       CoOp and original Tip-Adapter
src/p25/pixel_prompt.py  Shared additive border prompt
tests/                   Protocol, method, gradient, and data-recovery checks
setup.ps1                Windows dependency installation
run.ps1                  Project Python launcher
verify_clip.py           Environment and model check
```

The current public submission contains code and reproducibility instructions.
Datasets, environments, weights, caches, generated manifests, raw results,
figures, and report materials remain local under the current `.gitignore` rules.
The result tables above summarize completed local experiments; a fresh clone
must generate or separately obtain their underlying artifacts.

## Challenges and next steps

Documented challenges include differences from published data/training protocols,
interrupted dataset downloads, and checkpoint-selection bias toward one
resolution. Shared manifests support internal comparisons; download retries and
completed-run skipping support recovery; all four prompt strategies use the
same cross-resolution validation objective.

The report's revised plan prioritizes the following:

| Project week | Task | Responsible |
| --- | --- | --- |
| 7 | Complete the selected published-baseline reproduction and document protocol differences | Yubo, Wangyi |
| 8 | Evaluate frozen visual-prompt checkpoints on the held-out test set, including resolution 24 | Miller, Yubo |
| 9 | Analyze seed sensitivity, confusion matrices, and resolution–accuracy trade-offs; prepare figures | Wangyi, Miller |
| 10 | Finalize the report, reproducibility documentation, and presentation | All |

Further experiments should examine other shot budgets and ways to reduce the
native-resolution accuracy loss. ResNet-50 remains uncompleted original-proposal
scope and has no implementation/results in the reviewed repository; it is not
part of the midterm result tables above.

## References

Numbering matches the midterm report.

1. Radford et al. (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020). *ICML*. arXiv:2103.00020.
2. Zhou et al. (2022). [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134). *IJCV*. arXiv:2109.01134.
3. Zhang et al. (2022). [Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification](https://arxiv.org/abs/2207.09519). *ECCV*. arXiv:2207.09519.
4. Jia et al. (2022). [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119). *ECCV*. arXiv:2203.12119.
5. Bahng et al. (2022). [Exploring Visual Prompts for Adapting Large-Scale Models](https://arxiv.org/abs/2203.17274). *arXiv preprint*. arXiv:2203.17274.
6. Corley et al. (2023). [Revisiting Pre-trained Remote Sensing Model Benchmarks: Resizing and Normalization Matters](https://arxiv.org/abs/2305.13456). *arXiv preprint*. arXiv:2305.13456.
7. Wang et al. (2024). [A Hard-to-Beat Baseline for Training-free CLIP-based Adaptation](https://arxiv.org/abs/2402.04087). *ICLR*. arXiv:2402.04087.
8. Wu et al. (2024). [Unleashing the Power of Visual Prompting at the Pixel Level](https://arxiv.org/abs/2212.10556). *Transactions on Machine Learning Research (TMLR)*. arXiv:2212.10556.
9. Pathak et al. (2025). [LR0.FM: Low-Res Benchmark and Improving Robustness for Zero-Shot Classification in Foundation Models](https://arxiv.org/abs/2502.03950). *ICLR*. arXiv:2502.03950.

## AI assistance

OpenAI ChatGPT/Codex assisted with baseline code generation and integration,
experiment setup and debugging, verification scripts, figures, Excel analysis,
suggested interpretations, and report-text/LaTeX drafting and revision.

Reported numerical results come from executed experiments and saved predictions.
AI-assisted consistency checks compared predictions, confusion matrices,
metrics, sample manifests, and spreadsheet formulas. Qualitative examples use
original EuroSAT images with labels traced to saved predictions. Team members
remain responsible for reviewing the implementation, checking results and
references, and interpreting the findings.
