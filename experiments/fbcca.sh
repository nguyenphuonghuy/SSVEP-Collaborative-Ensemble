#!/bin/bash

PY=${PY:-python}
PROJ=${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}
export CUDA_MPS_PIPE_DIRECTORY=/dev/null    # BAT BUOC - fix MPS treo tren compute node
export FI_PROVIDER=tcp    # BAT BUOC - tranh "import torch" bi treo 55-185s cho dich vu dam phan CXI (Slingshot) tren node ai01
LOG=$PROJ/logs/fbcca_$(date +%Y%m%d_%H%M%S).log
mkdir -p $PROJ/logs
cd $PROJ
export PYTHONUNBUFFERED=1
{
echo "START $(date) node=$(hostname)"
echo "=== Novelty ablation: FBCCA baseline (training-free, CPU), @5.44s, 35 subject ==="
$PY -u evaluation/fbcca_baseline.py \
    --data_path data/raw/benchmark --n_subjects 35 \
    --t_start 0.14 --t_end 5.14 \
    --save_dir results/fbcca_fixedwindow_round
rc=$?
if [ $rc -ne 0 ]; then echo "!! FBCCA FAILED (rc=$rc)"; exit 1; fi
echo "DONE $(date)"
} > $LOG 2>&1
