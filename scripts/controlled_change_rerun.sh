#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
W=${WORKERS:-9}
OUT=outputs/cc
mkdir -p "$OUT"
COMMON=(--drones 10 --direction up --kind mixed --workers "$W")
CAMP=(--source feds --stages 6 40 80)
SYN=(--source synthetic --points 137)
run() { "$PY" scripts/run_change_experiment.py "$@" > /dev/null; }

final() {
    local R=$1
    run "${COMMON[@]}" "${CAMP[@]}" --trials 20 --first-seed 5000 --selective-radius "$R" --out "$OUT/camp.csv"
    run "${COMMON[@]}" "${SYN[@]}" --trials 30 --first-seed 5000 --selective-radius "$R" --out "$OUT/synthetic.csv"
    run "${COMMON[@]}" "${CAMP[@]}" --trials 20 --first-seed 5000 --selective-radius 1000 \
        --planners aco_selective_reset --out "$OUT/camp_selective_r1000.csv"
    run "${COMMON[@]}" "${CAMP[@]}" --trials 20 --first-seed 5000 --selective-radius "$R" \
        --params configs/aco_low_rho.json \
        --planners aco_retention aco_full_reset aco_fast_evaporation aco_selective_reset --out "$OUT/camp_low_rho.csv"
    for e in 4800 12000; do
        run "${COMMON[@]}" "${CAMP[@]}" --trials 10 --first-seed 5000 --planners vns --vns-evals "$e" \
            --out "$OUT/camp_vns$e.csv"
    done
    "$PY" scripts/run_change_experiment.py --drones 10 --direction cycle --kind mixed --workers "$W" "${CAMP[@]}" \
        --trials 10 --first-seed 5000 --selective-radius "$R" \
        --planners aco_retention aco_full_reset aco_selective_reset --out "$OUT/camp_cycle.csv" > /dev/null
}

budget() {
    for it in 240 600; do
        run "${COMMON[@]}" "${CAMP[@]}" --trials 10 --first-seed 5000 --ants 20 --iterations "$it" \
            --planners aco_full_reset aco_retention --out "$OUT/camp_aco$((20 * it)).csv"
    done
}

diagnostic() {
    "$PY" scripts/run_change_experiment.py --drones 10 --direction cycle --kind mixed --workers "$W" "${CAMP[@]}" \
        --trials 10 --first-seed 5000 --params configs/aco_untuned.json --ants 10 --iterations 15 \
        --planners aco_retention aco_full_reset aco_selective_reset --out "$OUT/camp_untuned_cycle.csv" > /dev/null
}

# Same design as diagnostic, but with the final tuned parameters (ACOParams defaults: rho = 0.1, 20 x 60).
tuned() {
    "$PY" scripts/run_change_experiment.py --drones 10 --direction cycle --kind mixed --workers "$W" "${CAMP[@]}" \
        --trials 10 --first-seed 5000 --ants 20 --iterations 60 \
        --planners aco_retention aco_full_reset aco_selective_reset --out "$OUT/camp_tuned_rho01_cycle.csv" > /dev/null
}

figures() {
    mkdir -p docs/figures
    for v in margin total_overdue; do
        planners=(aco_retention aco_fast_evaporation aco_selective_reset)
        [ "$v" = total_overdue ] && planners+=(vns)
        "$PY" scripts/plot_recovery.py "$OUT/camp.csv" "$OUT/camp_low_rho.csv" "$OUT/synthetic.csv" --value "best_$v" \
            --planners "${planners[@]}" \
            --titles "Camp Fire, tuned (rho = 0.36)" "Camp Fire, low rho (0.021)" "Synthetic, 137 points, tuned" \
            --out "docs/figures/controlled-change-recovery-$v.png" > /dev/null
    done
}

case "${1:-}" in
pilot)
    for r in 0 1000 2000 3000; do
        run "${COMMON[@]}" "${CAMP[@]}" --phase pilot --trials 10 --first-seed 4000 \
            --planners aco_selective_reset --selective-radius "$r" --out "$OUT/pilot_camp_r$r.csv"
        run "${COMMON[@]}" "${SYN[@]}" --phase pilot --trials 10 --first-seed 4000 \
            --planners aco_selective_reset --selective-radius "$r" --out "$OUT/pilot_syn_r$r.csv"
    done
    ;;
final) final "${2:?radius in metres, from the pilot}" ;;
budget) budget ;;
diagnostic) diagnostic ;;
tuned) tuned ;;
figures) figures ;;
all)
    final "${2:?radius in metres, from the pilot}"
    budget
    diagnostic
    figures
    ;;
*)
    echo "usage: $0 pilot | final RADIUS_M | budget | diagnostic | tuned | figures | all RADIUS_M" >&2
    exit 2
    ;;
esac
