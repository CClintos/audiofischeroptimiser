param([string]$PythonExe = "")

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
$python = if ($PythonExe) { (Resolve-Path -LiteralPath $PythonExe).Path } else { Join-Path $here ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment missing. Run .\setup_gui.ps1 or pass -PythonExe."
}
& $python -m PyInstaller --noconfirm --clean AudioFischerOptimizer.spec
if ($LASTEXITCODE -ne 0) { throw "GUI build failed." }
$exe = Join-Path $here "dist\AudioFischerOptimizer\AudioFischerOptimizer.exe"
if (-not (Test-Path -LiteralPath $exe)) { throw "Built executable was not found." }
Write-Output $exe
