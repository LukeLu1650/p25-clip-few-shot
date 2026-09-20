# Shared experiment configurations

Add validated configuration files here after the first pilot. No executable configuration is supplied yet.

Record for every experiment:
- Method, dataset version, split protocol, and class ordering.
- CLIP checkpoint (ViT-B/32), preprocessing, and training augmentations.
- Shots per class: 1, 2, 4, 8, or 16.
- Seed and shared training/validation sample manifest paths.
- Validation-label budget and hyperparameter search budget.
- Optimizer, learning rate, epochs, batch size, and checkpoint-selection rule where applicable.
- Prompt template, context length, initialization, or cache settings.
- Visual prompt resolution, amplitude bound, and regularization where applicable.
- Hardware, dependency versions, and output location.

Use repository-relative paths. Keep data and weights outside version control. Select settings using validation data only; reserve test labels for evaluation.
