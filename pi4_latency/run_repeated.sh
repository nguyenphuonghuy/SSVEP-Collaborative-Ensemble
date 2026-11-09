#!/bin/bash
# Chay latency_benchmark_pi.py N_RUNS lan doc lap (tien trinh rieng),
# nghi COOLDOWN giay giua cac lan de tranh thien vi do throttling nhiet
# tich luy tren Pi 4.
set -e
cd "$(dirname "$0")"
source ~/ssvep_latency_venv/bin/activate

N_RUNS=${1:-7}
COOLDOWN=${2:-15}
OUTDIR=runs
mkdir -p "$OUTDIR"

for i in $(seq 1 "$N_RUNS"); do
    echo "=== Run $i/$N_RUNS ==="
    python latency_benchmark_pi.py --out "$OUTDIR/run_${i}.json"
    if [ "$i" -lt "$N_RUNS" ]; then
        echo "-- cooldown ${COOLDOWN}s --"
        sleep "$COOLDOWN"
    fi
done

python aggregate_runs.py --pattern "$OUTDIR/run_*.json" --out pi4_latency_summary_${N_RUNS}runs.json
