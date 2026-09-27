param([Parameter(ValueFromRemainingArguments=$true)][string[]]$LeagueArgs)
$ErrorActionPreference = 'Stop'
$python = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $python = (Get-Command python -ErrorAction Stop).Source
}
& $python -u (Join-Path $PSScriptRoot 'league.py') @LeagueArgs
exit $LASTEXITCODE
