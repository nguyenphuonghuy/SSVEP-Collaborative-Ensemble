#!/bin/bash
# Usage: bash run_all.sh <python> [N_RUNS] [COOLDOWN]
# Interleaves configurations across repeated runs to spread thermal/frequency drift.
PY=$1; N=${2:-7}; CD=${3:-5}
cd "$(dirname "$0")"; mkdir -p runs
echo "START $(date) $(hostname)" > progress.log
$PY agree.py --out agree_results.json >> progress.log 2>&1
declare -A NS=( [1s]=250 [2s]=500 [4s]=1000 [5s]=1250 )
for r in $(seq 1 $N); do
  for w in 1s 2s 4s 5s; do
    T=${NS[$w]}; D=data/latency_trials_${w}.npz
    for cfg in "ensemble 1" "ensemble 4" "ensemble_dft 1" "ensemble_dft 4" "freqnet 1" "fbcca_cached 1" "fbcca_cached 4"; do
      set -- $cfg
      $PY bench.py --mode $1 --threads $2 --n_samples $T --data $D --out runs/${1}_${w}_t${2}_r${r}.json >> progress.log 2>&1 || echo "FAIL $cfg $w r$r" >> progress.log
      sleep $CD
    done
    if [ $r -le 3 ]; then
      $PY bench.py --mode fbcca_ref --threads 1 --n_samples $T --data $D --warmup 3 --iters 30 --out runs/fbcca_ref_${w}_t1_r${r}.json >> progress.log 2>&1 || echo "FAIL ref $w r$r" >> progress.log
      sleep $CD
    fi
  done
done
echo "DONE $(date)" >> progress.log
