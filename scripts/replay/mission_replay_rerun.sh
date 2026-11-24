#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
W=${WORKERS:-9}
OUT=outputs/mission
mkdir -p "$OUT" outputs/figures
COMMON=(--hours 24 --seeds 10 --first-seed 0 --workers "$W")
ACO=(aco_retention aco_full_reset aco_fast_evaporation aco_selective_reset)
run() { local name=$1; shift; "$PY" scripts/replay/aco_experiment.py "${COMMON[@]}" "$@" \
    --out "$OUT/$name.csv" --stages-out "$OUT/${name}_stages.csv" > "$OUT/$name.log"; }

run d10 --drones 10
run d6 --drones 6
run d10_widespread --drones 10 --trigger widespread --planners "${ACO[@]}"
"$PY" scripts/replay/plot_mission_accuracy.py "$OUT/d10_stages.csv" --out outputs/figures/mission-accuracy-d10.png > /dev/null
"$PY" scripts/replay/plot_mission_accuracy.py "$OUT/d6_stages.csv" --out outputs/figures/mission-accuracy-d6.png > /dev/null
