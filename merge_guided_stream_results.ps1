param(
    [string]$Root = "Optimizer_Component_20min_70cpu",
    [int]$Top = 20,
    [string]$DataRoot = "",
    [string]$Baseline = "",
    [string]$Target = "",
    [double]$ValidationThreshold = 2.5,
    [double]$GateMs = 0.0,
    [double]$SampleRate = 96000.0,
    [ValidateSet("auto", "off")]
    [string]$PhaseWrites = "auto"
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

function Quote-Arg([string]$Value) {
    if ($null -eq $Value) { return '""' }
    return '"' + ($Value -replace '(\\*)"', '$1$1\"') + '"'
}

function Join-Args([string[]]$Items) {
    return ($Items | ForEach-Object { Quote-Arg $_ }) -join ' '
}

$pythonExe = Join-Path $here ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Missing Python runtime at $pythonExe"
}
if (-not (Test-Path -LiteralPath $Root)) {
    throw "Run root not found: $Root"
}
$dataRootPath = if ($DataRoot -ne "") { $DataRoot } else { $here }
$baselinePath = if ($Baseline -ne "") { $Baseline } else { Join-Path $dataRootPath "baseline.afpx" }
$targetPath = if ($Target -ne "") { $Target } else { Join-Path $here "ResoNix Target Curve 2026.txt" }
if (-not (Test-Path -LiteralPath $baselinePath)) {
    throw "Baseline AFPX not found: $baselinePath"
}
if (-not (Test-Path -LiteralPath $targetPath)) {
    throw "Target curve not found: $targetPath"
}

$dataRootPath = (Resolve-Path -LiteralPath $dataRootPath).Path
$baselinePath = (Resolve-Path -LiteralPath $baselinePath).Path
$targetPath = (Resolve-Path -LiteralPath $targetPath).Path

$env:AFPX_DATA_ROOT = $dataRootPath
$env:AFPX_BASELINE = $baselinePath
$env:AFPX_TARGET = $targetPath

$args = @("_merge_stream_results.py", $Root, "--out", (Join-Path $Root "_merged_top"), "--top", "$Top", "--validation-threshold", "$ValidationThreshold")
if ($Baseline -ne "") { $args += @("--baseline", $baselinePath) }
if ($Target -ne "") { $args += @("--target", $targetPath) }
if ($GateMs -gt 0) { $args += @("--gate-ms", "$GateMs") }
$args += @("--sample-rate", "$SampleRate")
$args += @("--phase-writes", "$PhaseWrites")
$rootToken = "_merge_stream_results.py " + $Root
$existing = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -eq "python.exe" -and
        $_.CommandLine -like "*$rootToken*"
    } |
    Select-Object -First 1

if ($null -ne $existing) {
    Write-Host "Merge already running for $Root (PID $($existing.ProcessId))."
    exit 0
}

$activeWorkers = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -eq "python.exe" -and
        $_.CommandLine -like "*_optimizer_stream.py*" -and
        $_.CommandLine -like "*$Root\\worker_*"
    } |
    Select-Object -First 1

if ($null -ne $activeWorkers) {
    throw "Optimizer workers are still running for $Root (PID $($activeWorkers.ProcessId)). Merge after the run completes."
}

$argLine = Join-Args $args
$proc = Start-Process -FilePath $pythonExe `
    -WorkingDirectory $here `
    -ArgumentList $argLine `
    -NoNewWindow `
    -PassThru `
    -Wait

if ($proc.ExitCode -ne 0) {
    throw "_merge_stream_results.py exited with code $($proc.ExitCode)"
}
