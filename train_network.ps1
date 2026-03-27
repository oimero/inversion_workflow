param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ExperimentName
)

# This script is the PowerShell equivalent of train_network.sh.
# It creates a new experiment folder under experiments/results/<name>
# and starts experiments/main.py from that folder.

$startDir = (Get-Location).Path
$resultDir = Join-Path $startDir (Join-Path "experiments/results" $ExperimentName)

if (Test-Path $resultDir) {
    Write-Host "experiment folder already exists`t[aborting]"
    exit 1
}

Write-Host "Start directory: $startDir"
Write-Host "Create new experiment in $resultDir"
New-Item -Path $resultDir -ItemType Directory | Out-Null

Write-Host "Copy source and config files for reproducibility"
Copy-Item -Path (Join-Path $startDir "experiments/main.py") -Destination $resultDir
Copy-Item -Path (Join-Path $startDir "experiments/parameters.yaml") -Destination $resultDir

# Ensure local src package is importable after changing working directory.
$srcPath = Join-Path $startDir "src"
if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $env:PYTHONPATH = $srcPath
} else {
    $env:PYTHONPATH = "$srcPath;$env:PYTHONPATH"
}

Set-Location $resultDir
python main.py
