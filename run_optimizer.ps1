param(
    [string]$DataRoot = ".",
    [string]$Baseline = "",
    [string]$Target = "",
    [string]$Root = "",
    [int]$Seconds = 1200,
    [int]$Workers = 0,
    [ValidateSet("peq", "phase")]
    [string]$Mode = "peq",
    [ValidateSet("guided", "beam", "cmaes", "mixed", "random")]
    [string]$Proposal = "beam",
    [double]$ValidationThreshold = 2.5,
    [ValidateSet("auto", "off")]
    [string]$PhaseWrites = "auto",
    [string]$ImpulseRoot = "",
    [string]$LevelCalibration = "",
    [ValidateSet("off", "recommend")]
    [string]$SubBlend = "off",
    [double]$HeadroomDb = -1,
    [ValidateSet("off", "audition")]
    [string]$VoicingVariants = "off",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
$data = (Resolve-Path -LiteralPath $DataRoot).Path
$baselinePath = if ($Baseline) { (Resolve-Path -LiteralPath $Baseline).Path } else { Join-Path $data "baseline.afpx" }
$targetPath = if ($Target) { (Resolve-Path -LiteralPath $Target).Path } else { Join-Path $here "ResoNix Target Curve 2026.txt" }
$python = if ($PythonExe) { (Resolve-Path -LiteralPath $PythonExe).Path } else { Join-Path $here ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $python)) { throw "Python runtime not found: $python" }
if (-not (Test-Path -LiteralPath $baselinePath)) { throw "Baseline AFPX not found: $baselinePath" }
if (-not (Test-Path -LiteralPath $targetPath)) { throw "Target curve not found: $targetPath" }
$env:AFPX_DATA_ROOT = $data
$env:AFPX_BASELINE = $baselinePath
$env:AFPX_TARGET = $targetPath
if (-not $Root) { $Root = Join-Path $here ("Optimizer_Run_" + (Get-Date -Format "yyyyMMdd_HHmmss")) }
if ($Workers -le 0) {
    $logical = [Environment]::ProcessorCount
    $Workers = [Math]::Max(1, [Math]::Min(12, [Math]::Floor($logical * 0.60)))
}
if ($Mode -eq "phase") {
    $Workers = 1
    $Proposal = "beam"
    $PhaseWrites = "auto"
}
$launch = @{
    Root = $Root; Workers = $Workers; Seconds = $Seconds; Proposal = $Proposal
    Mode = $Mode
    DataRoot = $data; Baseline = $baselinePath; Target = $targetPath
    ValidationThreshold = $ValidationThreshold; PhaseWrites = $PhaseWrites
    ArchiveSize = 1200; Keep = 80; Top = 20; PythonExe = $python
}
if ($ImpulseRoot) { $launch.ImpulseRoot = $ImpulseRoot }
if ($LevelCalibration) { $launch.LevelCalibration = $LevelCalibration }
& (Join-Path $here "run_guided_stream_workers.ps1") @launch *> $null

$processFile = Join-Path $Root "worker_processes.json"
if (-not (Test-Path -LiteralPath $processFile)) { throw "Worker process manifest was not created." }
$workerRows = @((Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json) | ForEach-Object { $_ })
foreach ($row in $workerRows) {
    $process = Get-Process -Id ([int]$row.Id) -ErrorAction SilentlyContinue
    if ($process) { $process | Wait-Process }
}
$failed = @()
foreach ($row in $workerRows) {
    $state = Join-Path $row.Out "stream_state.json"
    $stderr = Join-Path $Root ($row.Worker + ".stderr.log")
    $rawError = if (Test-Path -LiteralPath $stderr) { Get-Content -LiteralPath $stderr -Raw } else { "" }
    $errorText = if ($null -eq $rawError) { "" } else { ([string]$rawError).Trim() }
    if (-not (Test-Path -LiteralPath $state) -or $errorText -match "Traceback|Error:|Exception") {
        $failed += "$($row.Worker): $errorText"
    }
}
if ($failed.Count) { throw ($failed -join [Environment]::NewLine) }

$phaseCache = Join-Path $Root "phase_diagnostics.json"
$mergeArgs = @(
    "_merge_stream_results.py", $Root, "--out", (Join-Path $Root "_merged_top"),
    "--top", "20", "--baseline", $baselinePath, "--target", $targetPath,
    "--validation-threshold", "$ValidationThreshold", "--phase-writes", $PhaseWrites
)
$mergeArgs += @("--sub-blend", $SubBlend, "--voicing-variants", $VoicingVariants)
$mergeArgs += @("--mode", $Mode)
if ($HeadroomDb -ge 0) { $mergeArgs += @("--headroom-db", "$HeadroomDb") }
if (Test-Path -LiteralPath $phaseCache) { $mergeArgs += @("--phase-cache", $phaseCache) }
if ($ImpulseRoot) { $mergeArgs += @("--impulse-root", $ImpulseRoot) }
if ($LevelCalibration) { $mergeArgs += @("--level-calibration", $LevelCalibration) }
$mergeOutput = & $python @mergeArgs 2>&1
if ($LASTEXITCODE -ne 0) { throw ($mergeOutput -join [Environment]::NewLine) }

$merged = Join-Path $Root "_merged_top"
$verifyDir = Join-Path $merged "verification"
New-Item -ItemType Directory -Force -Path $verifyDir | Out-Null
foreach ($candidate in Get-ChildItem -LiteralPath $merged -Filter "*.afpx" | Where-Object { $_.Name -like "family_*" -or $_.Name -like "voicing_*" }) {
    $verifyArgs = @(
        "scripts\verify_written_tune.py", $baselinePath, $candidate.FullName,
        "--allow-output-trim",
        "--out", (Join-Path $verifyDir ($candidate.BaseName + ".json"))
    )
    if ($PhaseWrites -eq "auto") { $verifyArgs += @("--allow-delay", "--allow-apf", "--allow-polarity") }
    $null = & $python @verifyArgs
    if ($LASTEXITCODE -ne 0) { throw "Candidate verification failed: $($candidate.Name)" }
}

$summary = (Resolve-Path -LiteralPath (Join-Path $merged "assistant_summary.json")).Path
Write-Output $summary
