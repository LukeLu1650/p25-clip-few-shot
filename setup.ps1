param(
    [string]$BootstrapPython = '',
    [ValidateSet('cu128', 'cpu')][string]$TorchBuild = 'cu128'
)
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Path (Join-Path $PSScriptRoot '.cache\pip') -Force | Out-Null
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot '.cache\pip'
if (-not (Test-Path -LiteralPath $projectPython)) {
    if ($BootstrapPython) { & $BootstrapPython -m venv (Join-Path $PSScriptRoot '.venv') }
    elseif (Get-Command py -ErrorAction SilentlyContinue) { & py -3.12 -m venv (Join-Path $PSScriptRoot '.venv') }
    else { throw 'Install Python 3.12 or pass -BootstrapPython with its executable path.' }
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
& $projectPython -m pip install 'setuptools<81' wheel
if ($LASTEXITCODE -ne 0) { throw 'Build dependency installation failed.' }
& $projectPython -m pip install torch==2.7.1 torchvision==0.22.1 --index-url "https://download.pytorch.org/whl/$TorchBuild"
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $projectPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Project dependency installation failed.' }
& $projectPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency compatibility check failed.' }
& $projectPython -m pip freeze | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'requirements.local.lock.txt') -Encoding utf8
$checkDevice = if ($TorchBuild -eq 'cpu') { 'cpu' } else { 'cuda' }
& $projectPython (Join-Path $PSScriptRoot 'verify_clip.py') --device $checkDevice
if ($LASTEXITCODE -ne 0) { throw 'CLIP verification failed.' }
