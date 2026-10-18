# Shared experimental protocol

## Data and label budgets

| Dataset | Fixed partition | Few-shot training | Validation labels used |
| --- | --- | --- | --- |
| EuroSAT RGB | Teammate script; 60/10/30, split seed 2026 | Exact shared index lists; 1/2/4/8/16 per class, seeds 1/2/3 | Entire 2,700-image validation partition |
| DTD | Official partition 1: 40/40/40 per class | Sample only official training pool | Fixed 4 per class, 188 total |
| Flowers102 | Custom per-class 50/20/30 over all 8,189 images; split seed 0 | Sample only custom training pool | Fixed 4 per class, 408 total |

EuroSAT has 16,200 pool images and 8,100 test images. The index manifest's fingerprint is checked against torchvision ordering before conversion to portable image paths. The runner does not resample EuroSAT. Keep the exact teammate JSON files and share them with all members. Class descriptions are mapped to readable text without changing numeric labels.

Flowers102's official training split has 10 examples/class and cannot support 16 shots without changing protocol. This custom split supports all five budgets and follows the **proportions** used in CoOp, but is not the published Zhou split. Its random lists differ. DTD's official partition also differs from CoOp's random partition. Do not compare these results directly to published numbers as if the partitions were identical. Give teammates the exported manifests; settle on one protocol before final runs.

Partitions are fixed independently of training seeds. Few-shot samples are nested across budgets within each seed. Native image paths and labels, class ordering, partition identity and sample identity are recorded. Validation labels are additional supervision, not part of the stated k-shot training budget. Zero-shot does not use training or validation labels and is evaluated only once per dataset.

## Model and preprocessing

- OpenAI CLIP ViT-B/32, official commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`.
- Official weight SHA-256: `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`; CLIP's loader verifies the downloaded checkpoint.
- FP32 backbone and features for baseline comparisons. Official CLIP resize, center crop and normalization; RGB input; no random image augmentation in these baseline implementations.
- Unit-normalized image and text features. Zero-shot and CoOp use the checkpoint logit scale; Tip-Adapter uses 100 times cosine similarity as in its reference implementation.
- Templates are predefined in `src/p25/data.py`; do not select them from test accuracy. Any `--template` variant is a separate experiment.

## Adaptation

CoOp learns one shared continuous context with class tokens at the end. CLIP parameters are frozen; gradients flow through its text encoder into context vectors. Deterministic frozen image features are cached. SGD momentum 0.9, cosine learning-rate schedule, default 50 epochs, lr 0.002, batch size 32, context length 16. Select the earliest best validation epoch. Initialization words must have exactly the requested number of BPE tokens; `a photo of a` is the provided four-token initialization.

This is a simplified CoOp baseline, not a claim to reproduce the original training recipe: no random crop/flip augmentation, no warmup, no class-specific/middle/front context variants. Do not label our scores as official paper reproductions. For strict reproduction, implement and compare the full upstream recipe under the team's shared samples.

Original Tip-Adapter uses normalized few-shot image features as fixed cache keys and one-hot labels as values. Logits are `100*cosine + alpha * exp(beta*(affinity-1)) @ values`. Search 6 alpha x 6 beta values using validation only. Alpha 0 is a valid no-cache candidate. No gradient training or Tip-Adapter-F is performed. The project uses one deterministic view per support image; upstream typically averages multiple augmented views. Record this difference.

The test partition is evaluated after model/parameter selection. Never pick an epoch, learning rate, template, context setting, alpha or beta based on test results. Predeclare the context sensitivity comparisons and report all of them.

## Reporting

Report top-1 accuracy in percent and sample standard deviation (`ddof=1`) over seeds 1, 2, 3. The aggregator includes scientific settings in the condition key, rejects duplicate seeds and flags fewer than three seeds. Do not treat repeated zero-shot evaluations as independent experiments. Keep pilot summaries separate from the main matrix.

Each completed run records configuration, manifest hash, class descriptions, label budgets, hardware, library versions, trainable parameter count, selected validation settings, predictions, confusion counts and text-to-image retrieval examples. A CoOp checkpoint stores context vectors only; Tip stores keys/values. Checkpoints and full raw runs stay local.

Timing separates model load, text encoding, feature extraction/cache reads, optimization and validation selection, and test evaluation. `cold_adaptation_seconds` uses original feature extraction durations even if cached, plus text encoding and selection/training. It excludes model download/load and test inference. Report that definition; do not compare a warm cache read with another method's full extraction time. Cache metadata assumes source images are immutable; clear the feature cache if replacing data files.

The optional `--resolutions 0 16 24 32 64` evaluates a fixed model on native and bicubically resized inputs. It does **not** train a multi-resolution visual prompt. Yubo/Miller must implement the submitted extension: matched single/native versus multi-resolution prompt training (16/32/64), unseen test resolution 24, same support lists, optimizer and training-update budgets. Visual-prompt training must operate on images with gradients through the frozen image encoder; the cached-feature CoOp path cannot train pixel prompts.
