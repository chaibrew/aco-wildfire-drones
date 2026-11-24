#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
W=${WORKERS:-9}
OUT=outputs/cc
mkdir -p "$OUT"
COMMON=(--drones 10 --direction cycle --kind mixed --workers "$W" --source feds --stages 6 40 80 --trials 10 --first-seed 5000)
run() { "$PY" scripts/controlled_change/run_change_experiment.py "${COMMON[@]}" "$@" > /dev/null; }

tuned() {
    run --ants 20 --iterations 60 --planners aco_retention aco_full_reset aco_selective_reset \
        --out "$OUT/camp_tuned_rho01_cycle.csv"
}

baselines() {
    run --ants 20 --iterations 60 --planners greedy vns --out "$OUT/camp_rho01_baselines.csv"
}

untuned() {
    run --params configs/aco_untuned.json --ants 10 --iterations 15 \
        --planners aco_retention aco_full_reset aco_selective_reset --out "$OUT/camp_untuned_cycle.csv"
}

figures() {
    mkdir -p outputs/figures
    for v in margin total_overdue; do
        "$PY" scripts/controlled_change/plot_recovery.py "$OUT/camp_tuned_rho01_cycle.csv" --value "best_$v" \
            --planners aco_retention aco_selective_reset --titles "Camp Fire, tuned" \
            --out "outputs/figures/controlled-change-recovery-$v.png" > /dev/null
    done
}

case "${1:-}" in
tuned) tuned ;;
baselines) baselines ;;
untuned) untuned ;;
figures) figures ;;
all)
    tuned
    baselines
    untuned
    figures
    ;;
*)
    echo "usage: $0 tuned | baselines | untuned | figures | all" >&2
    exit 2
    ;;
esac
