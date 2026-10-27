param(
    [Parameter(Mandatory = $true)][string]$SyntheaJar,
    [string]$WorkDir = (Join-Path $PSScriptRoot "work"),
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
$raw = Join-Path $WorkDir "synthea_raw"

java -Xmx8g -jar $SyntheaJar -p 1000 -s 20261012 -cs 20261012 -r 20261011 -e 20261011 `
    --exporter.baseDirectory="$raw" --exporter.csv.export=true --exporter.fhir.export=false

& $Python (Join-Path $PSScriptRoot "select_cohort_b.py") `
    --source (Join-Path $raw "csv") --output (Join-Path $WorkDir "input")

Push-Location $root
try {
    & $Python run_alignment_study.py `
        --input-dir (Join-Path $WorkDir "input") `
        --output-dir (Join-Path $WorkDir "results_reproduced")
} finally {
    Pop-Location
}

Write-Output "Completed cohort B regeneration under $WorkDir"
