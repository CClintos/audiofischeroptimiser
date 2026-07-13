$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $here ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "GUI environment missing. Run .\setup_gui.ps1 first."
}
Start-Process -FilePath $python -ArgumentList (Join-Path $here "audiofischer_gui.py") -WorkingDirectory $here
