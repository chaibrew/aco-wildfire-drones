#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
W=${WORKERS:-9}
OUT=outputs/cc
mkdir -p "$OUT"
COMMON=(--drones 10 --direction cycle --kind mixed --workers "$W" --source feds --stages 6 40 80 --trials 10 --first-seed 5000
    --ants 20 --iterations 60)
run() { "$PY" scripts/controlled_change/run_change_experiment.py "${COMMON[@]}" "$@" > /dev/null; }

tuned() {
    run --planners aco_retention aco_full_reset aco_selective_reset --out "$OUT/camp_tuned_rho01_cycle.csv"
}

baselines() {
    run --planners greedy vns --out "$OUT/camp_rho01_baselines.csv"
}

case "${1:-}" in
tuned) tuned ;;
baselines) baselines ;;
all)
    tuned
    baselines
    ;;
*)
    echo "usage: $0 tuned | baselines | all" >&2
    exit 2
    ;;
esac
