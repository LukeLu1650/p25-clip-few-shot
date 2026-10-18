# Experiments

- `pilot.json`: EuroSAT, one shot/class, one seed, two CoOp epochs; integration check only.
- `baselines.json`: 93 baseline jobs across three datasets, five budgets and three seeds.
- `coop_sensitivity.json`: 12 EuroSAT 16-shot jobs controlling context length and initialization.

Use `run.ps1 scripts/sweep.py --config configs/baselines.json --dry-run` to inspect the full plan. Extra runner flags are forwarded, for example `--device cpu` or `--download`. EuroSAT always imports the teammate's manifests (split seed 2026, full validation partition); JSON `split_seed` and `val_shots` options apply to other datasets. See `docs/PROTOCOL.md` before changing settings.
