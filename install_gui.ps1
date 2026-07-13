param(
    [string]$Source = "",
    [string]$Destination = ""
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourcePath = if ($Source) { (Resolve-Path -LiteralPath $Source).Path } else { Join-Path $here "dist\AudioFischerOptimizer" }
$destinationPath = if ($Destination) { [System.IO.Path]::GetFullPath($Destination) } else { Join-Path $env:LOCALAPPDATA "AudioFischerOptimizer" }
$allowedRoot = [System.IO.Path]::GetFullPath($env:LOCALAPPDATA).TrimEnd('\') + '\'
if (-not $destinationPath.StartsWith($allowedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Install destination must be inside LOCALAPPDATA: $allowedRoot"
}
if (-not (Test-Path -LiteralPath (Join-Path $sourcePath "AudioFischerOptimizer.exe"))) {
    throw "Built GUI package not found. Run .\build_gui.ps1 first."
}
if (Test-Path -LiteralPath $destinationPath) {
    Remove-Item -LiteralPath $destinationPath -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $destinationPath | Out-Null
Copy-Item -Path (Join-Path $sourcePath '*') -Destination $destinationPath -Recurse -Force

$shell = New-Object -ComObject WScript.Shell
$shortcutPath = Join-Path ([Environment]::GetFolderPath('Desktop')) "AudioFischer Optimizer.lnk"
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = Join-Path $destinationPath "AudioFischerOptimizer.exe"
$shortcut.WorkingDirectory = $destinationPath
$shortcut.Description = "Local AFPX tuning optimizer"
$shortcut.Save()

Write-Output (Join-Path $destinationPath "AudioFischerOptimizer.exe")
