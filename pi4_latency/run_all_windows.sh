#!/bin/bash
# Quet ca 4 do dai cua so (1s/2s/4s/5.44s), moi cua so chay N_RUNS lan doc
# lap (tien trinh rieng) de lay trung binh lien-tien-trinh, nghi COOLDOWN
# giay giua cac lan.
set -e
cd "$(dirname "$0")"
source ~/ssvep_latency_venv/bin/activate

N_RUNS=${1:-7}
COOLDOWN=${2:-15}

declare -A WINDOWS=( ["1s"]=250 ["2s"]=500 ["4s"]=1000 ["5s"]=1250 )

for label in "${!WINDOWS[@]}"; do
    n_samples=${WINDOWS[$label]}
    outdir="runs_window_${label}"
    mkdir -p "$outdir"
    echo "########## Window ${label} (n_samples=${n_samples}) ##########"
    for i in $(seq 1 "$N_RUNS"); do
        echo "=== ${label} run $i/$N_RUNS ==="
        python latency_benchmark_pi.py --n_samples "$n_samples" --out "$outdir/run_${i}.json"
        if [ "$i" -lt "$N_RUNS" ]; then
            echo "-- cooldown ${COOLDOWN}s --"
            sleep "$COOLDOWN"
        fi
    done
    python aggregate_runs.py --pattern "${outdir}/run_*.json" --out "pi4_latency_summary_${label}_${N_RUNS}runs.json"
    echo "-- cooldown ${COOLDOWN}s giua cac cua so --"
    sleep "$COOLDOWN"
done
