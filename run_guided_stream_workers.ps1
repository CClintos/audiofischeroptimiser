param(
    [string]$Root = "Optimizer_Component_20min_70cpu",
    [int]$Workers = 11,
    [int]$Seconds = 1200,
    [int]$StartSeed = 20260711,
    [int]$Top = 20,
    [int]$Keep = 80,
    [int]$ArchiveSize = 6000,
    [ValidateSet("guided", "random", "mixed", "cmaes", "beam")]
    [string]$Proposal = "guided",
    [double]$CmaSigma = 0.18,
    [int]$CmaPopulation = 0,
    [int]$BeamWidth = 24,
    [int]$BeamPoolLimit = 6,
    [double]$MaxPositiveGainPenalty = 0.0,
    [double]$ValidationThreshold = 2.5,
    [double]$GateMs = 0.0,
    [double]$SampleRate = 96000.0,
    [ValidateSet("auto", "off")]
    [string]$PhaseWrites = "auto",
    [string]$DataRoot = "",
    [string]$ImpulseRoot = "",
    [string]$LevelCalibration = "",
    [string]$Baseline = "",
    [string]$Target = "",
    [string]$PythonExe = ""
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

New-Item -ItemType Directory -Force -Path $Root | Out-Null

$pythonExe = if ($PythonExe -ne "") { $PythonExe } else { Join-Path $here ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Missing Python runtime at $pythonExe"
}

$dataRootPath = if ($DataRoot -ne "") { $DataRoot } else { $here }
$baselinePath = if ($Baseline -ne "") { $Baseline } else { Join-Path $dataRootPath "baseline.afpx" }
$targetPath = if ($Target -ne "") { $Target } else { Join-Path $here "ResoNix Target Curve 2026.txt" }
$impulseRootPath = if ($ImpulseRoot -ne "") { (Resolve-Path -LiteralPath $ImpulseRoot).Path } else { "" }
$levelCalibrationPath = if ($LevelCalibration -ne "") { (Resolve-Path -LiteralPath $LevelCalibration).Path } else { "" }
$phaseCachePath = Join-Path (Resolve-Path -LiteralPath $Root).Path "phase_diagnostics.json"
$stopFilePath = Join-Path (Resolve-Path -LiteralPath $Root).Path "stop_requested"
Remove-Item -LiteralPath $stopFilePath -Force -ErrorAction SilentlyContinue

if (-not (Test-Path -LiteralPath $baselinePath)) {
    throw "Baseline AFPX not found: $baselinePath"
}
if (-not (Test-Path -LiteralPath $targetPath)) {
    throw "Target curve not found: $targetPath"
}
if (-not (Test-Path -LiteralPath $dataRootPath)) {
    throw "Measurement folder not found: $dataRootPath"
}

$dataRootPath = (Resolve-Path -LiteralPath $dataRootPath).Path
$baselinePath = (Resolve-Path -LiteralPath $baselinePath).Path
$targetPath = (Resolve-Path -LiteralPath $targetPath).Path

$env:AFPX_DATA_ROOT = $dataRootPath
$env:AFPX_BASELINE = $baselinePath
$env:AFPX_TARGET = $targetPath

$cacheArgs = @(
    "scripts\prepare_phase_cache.py",
    "--data-root", $dataRootPath,
    "--baseline", $baselinePath,
    "--target", $targetPath,
    "--out", $phaseCachePath,
    "--validation-threshold", "$ValidationThreshold",
    "--print-mode", "none"
)
if ($impulseRootPath -ne "") { $cacheArgs += @("--impulse-root", $impulseRootPath) }
if ($levelCalibrationPath -ne "") { $cacheArgs += @("--level-calibration", $levelCalibrationPath) }
& $pythonExe @cacheArgs
if ($LASTEXITCODE -ne 0) { throw "Phase diagnostic cache preparation failed." }

$started = @()
for ($i = 1; $i -le $Workers; $i++) {
    $name = "worker_{0:D2}" -f $i
    $out = Join-Path $Root $name
    $stdout = Join-Path $Root ($name + ".stdout.log")
    $stderr = Join-Path $Root ($name + ".stderr.log")
    $seed = $StartSeed + $i

    $args = @(
        "_optimizer_stream.py",
        "--baseline", $baselinePath,
        "--target", $targetPath,
        "--seconds", "$Seconds",
        "--top", "$Top",
        "--keep", "$Keep",
        "--archive-size", "$ArchiveSize",
        "--proposal", "$Proposal",
        "--profile", "explore",
        "--filter-cost-scale", "0.1",
        "--min-total-bands", "0",
        "--cma-sigma", "$CmaSigma",
        "--cma-population", "$CmaPopulation",
        "--beam-width", "$BeamWidth",
        "--beam-pool-limit", "$BeamPoolLimit",
        "--max-positive-gain-penalty", "$MaxPositiveGainPenalty",
        "--validation-threshold", "$ValidationThreshold",
        "--sample-rate", "$SampleRate",
        "--phase-writes", "$PhaseWrites",
        "--phase-cache", $phaseCachePath,
        "--checkpoint-seconds", "60",
        "--stop-file", $stopFilePath,
        "--seed", "$seed",
        "--resume",
        "--out", $out
    )
    if ($GateMs -gt 0) {
        $args += @("--gate-ms", "$GateMs")
    }
    if ($impulseRootPath -ne "") {
        $args += @("--impulse-root", $impulseRootPath)
    }
    if ($levelCalibrationPath -ne "") {
        $args += @("--level-calibration", $levelCalibrationPath)
    }

    $argLine = Join-Args $args
    $p = Start-Process -FilePath $pythonExe `
        -WorkingDirectory $here `
        -ArgumentList $argLine `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru

    $started += [pscustomobject]@{
        Worker = $name
        Id = $p.Id
        Seed = $seed
        Out = $out
    }
}

Start-Sleep -Milliseconds 2000

$failed = @()
foreach ($item in $started) {
    $proc = Get-Process -Id $item.Id -ErrorAction SilentlyContinue
    if ($null -ne $proc) {
        continue
    }
    $stderr = Join-Path $Root ($item.Worker + ".stderr.log")
    $stdout = Join-Path $Root ($item.Worker + ".stdout.log")
    $errText = if (Test-Path -LiteralPath $stderr) { (Get-Content -LiteralPath $stderr -Raw -ErrorAction SilentlyContinue).Trim() } else { "" }
    $outText = if (Test-Path -LiteralPath $stdout) { (Get-Content -LiteralPath $stdout -Raw -ErrorAction SilentlyContinue).Trim() } else { "" }
    $failed += [pscustomobject]@{
        Worker = $item.Worker
        Id = $item.Id
        ErrorText = if ($errText) { $errText } elseif ($outText) { $outText } else { "worker exited immediately with no log output" }
    }
}

if ($failed.Count -gt 0) {
    $failed | Format-Table -AutoSize
    throw "One or more optimizer workers exited immediately. See the worker stderr logs above."
}

$started | Format-Table -AutoSize
$started | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Root "worker_processes.json") -Encoding UTF8
Write-Host ""
Write-Host "Started $Workers guided streaming workers for $Seconds seconds."
Write-Host "Run this same script again with the same -Root to resume/continue."
