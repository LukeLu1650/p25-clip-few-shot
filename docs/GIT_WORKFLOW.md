# One-folder Git workflow

Develop, run experiments and submit from this project folder. Local `main` tracks
`origin/main` in `LukeLu1650/p25-clip-few-shot`. A separate working copy is not needed.

## Current code-only submission

The current ignore rules exclude datasets, environments, caches, weights, local
settings, generated manifests, every results directory, historical/course
materials and reports. Three documentation files are allowed: this workflow,
PROTOCOL.md and UPSTREAM.md. The README describes this submission scope.

From the project root, inspect and stage:

```powershell
git status --short --branch
git add .
git diff --cached --stat
git diff --cached --check
```

Review the staged filenames. They should contain code, configs, tests, dependency
files, startup scripts, README, .gitignore and the three allowed documentation
files. They should not contain data, results, manifests, weights, personal settings
or historical documents. If unsure, inspect before proceeding.

Then execute the commit and push yourself:

```powershell
git commit -m "Add CLIP adaptation baselines and experiment tools"
git push origin main
```

If Git is not on PATH, invoke its executable by full path using the variable shown
in the chat. If a push is rejected, inspect its reported reason rather than using
force-push.

## Later result submission

Results and generated manifests remain locally available. After reviewing the
scientific evidence and document contents, deliberately narrow the corresponding
ignore rules or force-add only the specific reviewed files for a separate commit.
Do not force-add entire dataset, results or documentation directories. Publish
the selected summaries, figures and shared manifests with their protocol and
provenance; retain raw model/data artifacts outside version control.

Ignore rules do not remove already tracked files. Check both staged and tracked
filenames when changing submission scope. During setup, the remote history tracked
only .gitignore and three README files; no datasets or results were tracked.

## Ongoing collaboration

Inspect status and preserve intended work before integrating teammate updates.
Fetch and review incoming changes; resolve conflicts deliberately. Git setup and
documentation changes do not require rerunning scientifically unchanged experiments.
