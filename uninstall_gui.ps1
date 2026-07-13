$ErrorActionPreference = "Stop"
$destination = [System.IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "AudioFischerOptimizer"))
$allowedRoot = [System.IO.Path]::GetFullPath($env:LOCALAPPDATA).TrimEnd('\') + '\'
if (-not $destination.StartsWith($allowedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to remove a path outside LOCALAPPDATA."
}
$shortcut = Join-Path ([Environment]::GetFolderPath('Desktop')) "AudioFischer Optimizer.lnk"
Remove-Item -LiteralPath $shortcut -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $destination -Recurse -Force -ErrorAction SilentlyContinue
Write-Output "AudioFischer Optimizer removed."
