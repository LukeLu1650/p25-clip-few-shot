# Published-baseline validation: CLIP zero-shot on DTD

## Scope and predeclared target

This is a separate numerical validation of the zero-shot baseline, not a reproduction of CoOp or Tip-Adapter training. It does not replace the project's shared-protocol baseline and sensitivity results.

The target was fixed before evaluation: Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, Table 11, DTD, OpenAI CLIP ViT-B/32, **44.5%** top-1 accuracy. The local paper places this table on PDF page 43. The published value is rounded to one decimal place.

- Paper: https://proceedings.mlr.press/v139/radford21a.html
- Official class names and eight DTD templates: https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/data/prompts.md#describabletextures
- Official feature-ensemble procedure: https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/notebooks/Prompt_Engineering_for_ImageNet.ipynb
- Historical first-fold dataset convention: https://github.com/tensorflow/datasets/blob/v4.0.1/tensorflow_datasets/image_classification/dtd.py

The configuration preserves the official prompt metadata and its source SHA-256. The evaluator is project code; no complete upstream evaluator or trainer is vendored. OpenAI CLIP remains an external dependency with its existing MIT license.

## Fixed protocol

| Item | Setting |
| --- | --- |
| Checkpoint | Official OpenAI ViT-B/32; SHA-256 recorded and verified |
| Dataset | DTD RGB, official partition 1, `labels/test1.txt` |
| Test set | 1,880 images, exactly 40 per class, 47 classes |
| Train/validation label usage | Zero / zero |
| Training / optimizer / epochs | Not applicable: no adaptation |
| Text | Eight official DTD templates; normalize each embedding, average per class, normalize again |
| Image transform | Official CLIP resize, center crop, RGB conversion and normalization; no random augmentation |
| Precision | Official loader default: native mixed FP16/FP32 on CUDA; FP32 on CPU |
| Evaluation | Overall top-1 accuracy; one fixed evaluation, no prompt or hyperparameter selection using test results |

The CLIP paper does not explicitly identify its DTD fold. We use official fold 1, as used by the historical TensorFlow Datasets DTD implementation, and record the exact test-list hash. This is an explicitly documented protocol assumption, not proof that the original authors used an identical software environment and test-list file. PyTorch, torchvision and Pillow versions also differ from the original environment. Numerical agreement alone does not establish complete protocol identity.

This experiment differs from the project's existing DTD zero-shot condition: that condition uses one template and FP32. Keep both experiments labeled separately; do not replace an existing summary entry or attribute the difference only to ensembling.

## Run

From the project root, after the normal environment setup:

```powershell
.\run.ps1 scripts/reproduce_clip_dtd.py --download
```

The existing `run.ps1` selects the configured Python environment and model cache. Existing local DTD images and checkpoint are reused. On a new machine, `--download` allows dataset download and CLIP's loader downloads its checkpoint if needed. No new Python packages are required.

Completed outputs are protected against replacement. To perform a fresh repeat, use a new output directory:

```powershell
.\run.ps1 scripts/reproduce_clip_dtd.py --output-dir results/reproduction/clip_dtd_vit_b32_repeat
```

The default config is `configs/reproduction_clip_dtd.json`. It belongs to this standalone evaluator, not `scripts/sweep.py`. Results are stored under `results/reproduction/clip_dtd_vit_b32/`, outside `results/raw/`, so the main baseline aggregator does not combine the two protocols.

## Observed result

| Published | Measured | Correct / total | Absolute gap | Relative error |
| --- | --- | --- | --- | --- |
| 44.5% | 44.255319% | 832 / 1,880 | 0.244681 percentage points | 0.549845% |

The result falls within both 1 percentage point and 1% relative error of the rounded reference. No alternative template sets, folds or precisions were tried after observing this score.

The recorded run used an RTX 5070 Ti Laptop GPU, the official CLIP loader's native CUDA precision, PyTorch 2.7.1+cu128 and torchvision 0.22.1+cu128. The current evaluator and configuration match the SHA-256 hashes in the saved `metrics.json`. Its `model_dtype` field records the first parameter's dtype, not every parameter's dtype; CLIP uses a mixture of FP16 and FP32 parameters on CUDA.

Since there is no support sampling or optimization, report this fixed zero-shot check as **one evaluation, standard deviation not applicable**. Three identical repeats would not constitute three independent training/sampling runs. Continue to report three independent seeds for the adapted methods. This check does not establish that all midterm reproduction requirements for the adapted methods are satisfied.

## Checks and output files

Run the lightweight checks without downloading data or loading CLIP weights:

```powershell
.\run.ps1 -B -m unittest discover -s tests -p test_reproduction_clip_dtd.py -v
```

The three tests cover prompt-feature normalization before averaging, percentage-point versus relative-error calculations, and rejection of reordered classes or duplicate test images. They do not replace a full DTD evaluation.

A completed evaluation writes these files in the selected output directory:

- `metrics.json`: accuracy, environment, configuration and hashes.
- `comparison.csv`: measured accuracy and difference from the reference.
- `test_manifest.json`: evaluated image paths and labels.
- `predictions.csv`: per-image labels and predictions.
- `confusion.json`: confusion counts for the 47 classes.

Generated results remain local under the repository's existing ignore rules. The source files and this document allow teammates to run the same evaluation; the dataset, checkpoint and cache are not included.

## Suggested report text

> As a numerical validation of our zero-shot implementation, we evaluated OpenAI CLIP ViT-B/32 on the 1,880-image first-fold DTD test set using the eight official DTD text templates and the authors' normalized feature-ensemble procedure. Without using training or validation labels, our implementation achieved 44.26% top-1 accuracy, compared with 44.5% in CLIP Table 11, an absolute difference of 0.24 percentage points. We use the historical TFDS first-fold convention because the paper does not explicitly name its DTD fold; modern software versions also differ. This validation is separate from our shared-protocol CoOp and Tip-Adapter experiments and does not establish strict reproduction of those adapted methods.

The original teammate EuroSAT split script and existing baseline results are unchanged. AI assistance was used to prepare the evaluator, checks and documentation; the reported score was measured by executing the evaluator on the local dataset, not generated or estimated.
