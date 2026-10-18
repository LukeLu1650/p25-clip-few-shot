$ErrorActionPreference = 'Stop'
$checkArgs = @('verify_clip.py') + $args
$projectSettings = Join-Path $PSScriptRoot 'local-settings.json'
if (Test-Path -LiteralPath $projectSettings) {
    $localOptions = Get-Content -LiteralPath $projectSettings -Raw | ConvertFrom-Json
    if ($localOptions.clip_cache -and '--clip-cache' -notin $args) { $checkArgs += @('--clip-cache', $localOptions.clip_cache) }
}
& (Join-Path $PSScriptRoot 'run.ps1') @checkArgs
