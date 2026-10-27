#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 /path/to/synthea-4.0.1-SNAPSHOT-with-dependencies.jar [workdir]" >&2
    exit 2
fi

JAR="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
WORK="${2:-$HERE/work}"
mkdir -p "$WORK"

java -Xmx8g -jar "$JAR" -p 1000 -s 20261012 -cs 20261012 -r 20261011 -e 20261011 \
    --exporter.baseDirectory="$WORK/synthea_raw" \
    --exporter.csv.export=true --exporter.fhir.export=false

python "$HERE/select_cohort_b.py" --source "$WORK/synthea_raw/csv" --output "$WORK/input"

cd "$ROOT"
python run_alignment_study.py --input-dir "$WORK/input" --output-dir "$WORK/results_reproduced"

echo "Completed cohort B regeneration under $WORK"
