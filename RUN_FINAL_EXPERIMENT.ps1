param([string]$OutputDir = "reproduction_output")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$inputDir = Join-Path $root "data\synthea_input\csv"
$outDir = if ([System.IO.Path]::IsPathRooted($OutputDir)) { $OutputDir } else { Join-Path $root $OutputDir }

function Invoke-PythonStep {
    & python -B @args
    if ($LASTEXITCODE -ne 0) {
        throw "Python step failed (exit $LASTEXITCODE): $($args -join ' ')"
    }
}

Push-Location -LiteralPath $root
try {
    if (Test-Path -LiteralPath $outDir) {
        if ((Get-ChildItem -LiteralPath $outDir -Force | Measure-Object).Count -gt 0) {
            throw "Output directory is not empty: $outDir. Specify a new directory with -OutputDir."
        }
    }
    Invoke-PythonStep verify_input_hashes.py $inputDir
    Invoke-PythonStep -m unittest -q test_synthea_pipeline.py
    Invoke-PythonStep run_alignment_study.py --input-dir $inputDir --output-dir $outDir
    Invoke-PythonStep audit_alignment_study.py --study-dir $outDir --input-dir $inputDir
    Invoke-PythonStep plot_alignment_study.py --study-dir $outDir --output-dir (Join-Path $outDir "figures")
    Invoke-PythonStep audit_inferred_clusters.py --study-dir $outDir --output-dir (Join-Path $outDir "inferred_cluster_audit")
    Invoke-PythonStep run_oracle_weight_diagnostic.py --study-dir $outDir --output-dir (Join-Path $outDir "oracle_weight_diagnostic")
    Invoke-PythonStep supplemental_sensitivity.py --input-dir $inputDir --output-dir (Join-Path $outDir "supplemental_sensitivity")
    Invoke-PythonStep check_results.py --study-dir $outDir
    Write-Host "Completed final-paper reproduction in $outDir"
} finally {
    Pop-Location
}
