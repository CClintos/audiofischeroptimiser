param(
    [string]$Root = "Optimizer_Component_20min_70cpu",
    [int]$Workers = 11,
    [int]$Seconds = 1200,
    [int]$StartSeed = 2026070800,
    [int]$Top = 20,
    [int]$Keep = 80,
    [int]$ArchiveSize = 6000,
    [ValidateSet("guided", "random", "mixed", "cmaes")]
    [string]$Proposal = "guided",
    [double]$CmaSigma = 0.18,
    [int]$CmaPopulation = 0,
    [double]$MaxPositiveGainPenalty = 0.0,
    [double]$ValidationThreshold = 2.5,
    [double]$GateMs = 0.0,
    [string]$DataRoot = "",
    [string]$Baseline = "",
    [string]$Target = ""
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

$pythonExe = Join-Path $here ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Missing Python runtime at $pythonExe"
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
if (-not (Test-Path -LiteralPath $dataRootPath)) {
    throw "Measurement folder not found: $dataRootPath"
}

$dataRootPath = (Resolve-Path -LiteralPath $dataRootPath).Path
$baselinePath = (Resolve-Path -LiteralPath $baselinePath).Path
$targetPath = (Resolve-Path -LiteralPath $targetPath).Path

$env:AFPX_DATA_ROOT = $dataRootPath
$env:AFPX_BASELINE = $baselinePath
$env:AFPX_TARGET = $targetPath

$measurementNames = @(
    @("System Sum.txt", "SYSTEM SUM.txt"),
    @("Sub.txt", "SUB.txt"),
    @("Front L High.txt", "Front L Tweeter.txt"),
    @("Front R High.txt", "Front R Tweeter.txt"),
    @("Front L Low.txt", "Front L Mid.txt", "Front L MID.txt"),
    @("Front R Low.txt", "Front R Mid.txt", "Front R MID.txt"),
    @("Tweeters Together.txt", "Both Tweeters.txt"),
    @("Mid Bass Together.txt", "Both Mids.txt")
)
foreach ($aliases in $measurementNames) {
    $found = $false
    foreach ($name in $aliases) {
        if (Test-Path -LiteralPath (Join-Path $dataRootPath $name)) {
            $found = $true
            break
        }
    }
    if (-not $found) {
        throw "Missing required measurement file. Tried: $($aliases -join ', ') in $dataRootPath"
    }
}

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
        "--max-positive-gain-penalty", "$MaxPositiveGainPenalty",
        "--validation-threshold", "$ValidationThreshold",
        "--checkpoint-seconds", "60",
        "--seed", "$seed",
        "--resume",
        "--out", $out
    )
    if ($GateMs -gt 0) {
        $args += @("--gate-ms", "$GateMs")
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
Write-Host ""
Write-Host "Started $Workers guided streaming workers for $Seconds seconds."
Write-Host "Run this same script again with the same -Root to resume/continue."
