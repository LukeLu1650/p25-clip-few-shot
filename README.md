# P25: Few-Shot CLIP Adaptation

CISC 473 Deep Learning Capstone — Fall 2026

Compare efficient CLIP adaptation methods on **EuroSAT**, **DTD**, and **Oxford Flowers102**, and investigate a pixel-space visual-prompt extension on EuroSAT.

**Status:** planning and proposal stage. Training code, validated dependencies, and experimental results are not yet available. The visual-prompt extension is a research hypothesis, not a verified novel method.

## Planned methods

| Method | Adaptation |
| --- | --- |
| Zero-shot CLIP (ViT-B/32) | Fixed class descriptions; no target-task training |
| CoOp | Learn text-context vectors with CLIP frozen |
| Tip-Adapter | Build a labelled feature cache; distinguish the original method from Tip-Adapter-F |
| Supervised ResNet-50 | Fine-tune an ImageNet-pretrained model on matched few-shot samples |
| Pixel-space visual prompting | Reproduce an existing image-prompt baseline, then investigate prompt resolution and regularization |

## Experimental protocol

- Evaluate required few-shot methods at **1, 2, 4, 8, and 16 examples per class**.
- Share the same CLIP checkpoint, class mappings, fixed train/validation/test partitions, and sampled training lists across methods.
- Document each dataset's split protocol, particularly Flowers102. Report validation-label budgets separately.
- Use at least three seeds for stochastic experiments; report mean and standard deviation. Do not tune on test results.
- Report accuracy, trainable parameters, adaptation time, and hardware. Include cache construction and tuning costs where relevant.
- Concentrate the proposed visual-prompt extension and detailed ablations on EuroSAT.
- Treat frozen-feature extraction and prompt training separately: visual-prompt training still requires gradients through the image encoder.

## Team responsibilities

Replace A/B/C with agreed member names. Do not publish student IDs in this public repository.

| Member | Primary responsibilities |
| --- | --- |
| A — Project Lead | Data splits, evaluation, zero-shot CLIP, ResNet-50, integration |
| B — Research Lead | Literature review, CoOp, text-prompt sensitivity |
| C — Engineering Lead | Tip-Adapter, visual-prompt prototype, reproducibility |

After the visual-prompt prototype works, A leads the conventional pixel-prompt baseline, B the low-resolution unregularized variant, and C the regularized variant using shared code. Everyone contributes to repeated experiments, analysis, writing, and presentation.

## Repository contents

- `P25_proposal.tex`: working English LaTeX proposal draft; review before submission. It may lag behind edits made in Overleaf.
- `src/`: implementation workspace and guidance.
- `configs/`: shared experiment configuration guidance.
- `.gitignore`: excludes local data, feature caches, model weights, secrets, and temporary outputs.

## References

1. Radford et al. (2021), [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020).
2. Zhou et al. (2022), [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134).
3. Zhang et al. (2022), [Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification](https://arxiv.org/abs/2207.09519).
4. Jia et al. (2022), [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119).
5. Bahng et al. (2022), [Exploring Visual Prompts for Adapting Large-Scale Models](https://arxiv.org/abs/2203.17274).

Pixel-space visual prompting has already been evaluated on EuroSAT. Applying it to that dataset alone is not a new contribution.

## AI usage

ChatGPT assisted with background explanations, proposal drafting and formatting, and initial repository organization. The team will verify claims, implement and run experiments, and write its own analysis. Update this disclosure to reflect actual use.
