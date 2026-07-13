param([string]$PythonExe = "")

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
if (-not $PythonExe) {
    $PythonExe = (Get-Command py -ErrorAction Stop).Source
    & $PythonExe -3 -m venv .venv
} else {
    & $PythonExe -m venv .venv
}
$venvPython = Join-Path $here ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements-gui.lock.txt
Write-Output "GUI environment ready. Run .\start_gui.ps1"
