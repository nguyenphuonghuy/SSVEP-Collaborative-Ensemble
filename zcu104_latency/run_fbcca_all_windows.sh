#!/bin/bash
# Quet FBCCA latency qua 4 do dai cua so, moi cua so N_RUNS lan doc lap.
set -e
cd "$(dirname "$0")"
source ~/ssvep_zcu104_pkg/venv/bin/activate

N_RUNS=${1:-7}
COOLDOWN=${2:-10}

declare -A WINDOWS=( ["1s"]=250 ["2s"]=500 ["4s"]=1000 ["5s"]=1250 )

for label in "${!WINDOWS[@]}"; do
    n_samples=${WINDOWS[$label]}
    outdir="fbcca_runs_window_${label}"
    mkdir -p "$outdir"
    echo "########## FBCCA window ${label} (n_samples=${n_samples}) ##########"
    for i in $(seq 1 "$N_RUNS"); do
        echo "=== ${label} run $i/$N_RUNS ==="
        python fbcca_latency_zcu104.py --n_samples "$n_samples" --out "$outdir/run_${i}.json"
        if [ "$i" -lt "$N_RUNS" ]; then
            sleep "$COOLDOWN"
        fi
    done
    python aggregate_runs.py --pattern "${outdir}/run_*.json" --out "fbcca_latency_summary_${label}_${N_RUNS}runs.json"
    sleep "$COOLDOWN"
done
