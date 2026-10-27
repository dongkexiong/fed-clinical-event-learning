#!/usr/bin/env bash
# Run the frozen final-paper experiment from any working directory.
set -Eeuo pipefail
trap 'status=$?; printf "ERROR: step failed (exit %s), line %s.\n" "$status" "$LINENO" >&2; exit "$status"' ERR

usage() {
    printf 'Usage: bash RUN_FINAL_EXPERIMENT.sh [OUTPUT_DIR]\n'
    printf 'Relative output paths are resolved against the package directory.\n'
    printf 'Default: reproduction_output. Output must be absent or empty.\n'
    printf 'Set PYTHON to a Python executable (default: python3).\n'
}
if [[ $# -eq 1 && ( $1 == --help || $1 == -h ) ]]; then
    usage
    exit 0
fi
if [[ $# -gt 1 || ( $# -eq 1 && -z $1 ) ]]; then
    usage >&2
    exit 2
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd -- "$root"
input_dir="$root/data/synthea_input/csv"
out_dir=${1:-reproduction_output}
if [[ $out_dir != /* ]]; then
    out_dir="$root/$out_dir"
fi
python_bin=${PYTHON:-python3}
if ! command -v -- "$python_bin" >/dev/null 2>&1; then
    printf 'ERROR: Python executable not found: %s\n' "$python_bin" >&2
    exit 127
fi
if [[ -e $out_dir || -L $out_dir ]]; then
    if [[ ! -d $out_dir ]]; then
        printf 'ERROR: Output path is not a directory: %s\n' "$out_dir" >&2
        exit 1
    fi
    first_entry=$(find "$out_dir" -mindepth 1 -maxdepth 1 -print -quit)
    if [[ -n $first_entry ]]; then
        printf 'ERROR: Output directory is not empty: %s. Specify a new directory.\n' "$out_dir" >&2
        exit 1
    fi
fi

"$python_bin" -B verify_input_hashes.py "$input_dir"
"$python_bin" -B -m unittest -q test_synthea_pipeline.py
"$python_bin" -B run_alignment_study.py --input-dir "$input_dir" --output-dir "$out_dir"
"$python_bin" -B audit_alignment_study.py --study-dir "$out_dir" --input-dir "$input_dir"
"$python_bin" -B plot_alignment_study.py --study-dir "$out_dir" --output-dir "$out_dir/figures"
"$python_bin" -B audit_inferred_clusters.py --study-dir "$out_dir" --output-dir "$out_dir/inferred_cluster_audit"
"$python_bin" -B run_oracle_weight_diagnostic.py --study-dir "$out_dir" --output-dir "$out_dir/oracle_weight_diagnostic"
"$python_bin" -B supplemental_sensitivity.py --input-dir "$input_dir" --output-dir "$out_dir/supplemental_sensitivity"
"$python_bin" -B check_results.py --study-dir "$out_dir"
printf 'Completed final-paper reproduction in %s\n' "$out_dir"
