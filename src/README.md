# Implementation

`make_eurosat_splits.py` is the unchanged teammate script. `p25/data.py` validates and imports its exact index lists, handles the documented DTD/Flowers102 protocols and exports portable manifests. `p25/methods.py` implements shared-context CoOp and original training-free Tip-Adapter.

Run commands from the repository root through `run.ps1` or Python. See `docs/PROTOCOL.md` for differences from paper recipes and the interface requirements for teammates. ResNet-50 and image-level visual-prompt training remain to be added.
