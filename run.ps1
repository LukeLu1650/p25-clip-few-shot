# Invoke from any directory: .\run.ps1 scripts/run_experiment.py --dataset eurosat ...
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$projectSettings = Join-Path $PSScriptRoot 'local-settings.json'
if (Test-Path -LiteralPath $projectSettings) {
    $localOptions = Get-Content -LiteralPath $projectSettings -Raw | ConvertFrom-Json
    if (-not (Test-Path -LiteralPath $projectPython) -and $localOptions.python) { $projectPython = $localOptions.python }
    if ($localOptions.clip_cache) { $env:P25_CLIP_CACHE = $localOptions.clip_cache }
}
if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Run setup.ps1 first; Python environment not found.' }
Push-Location -LiteralPath $PSScriptRoot
try {
    & $projectPython @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
