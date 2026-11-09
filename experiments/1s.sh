#!/bin/bash

PY=${PY:-python}
PROJ=${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}
export CUDA_MPS_PIPE_DIRECTORY=/dev/null    # BAT BUOC - fix MPS treo tren compute node
export FI_PROVIDER=tcp    # BAT BUOC - tranh "import torch" bi treo 55-185s cho dich vu dam phan CXI (Slingshot) tren node ai01
LOG=$PROJ/logs/run_1s_$(date +%Y%m%d_%H%M%S).log
mkdir -p $PROJ/logs

cd $PROJ

# compute node không có internet -> tắt wandb để khỏi treo
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1

{
echo "════════ JOB START ════════"
echo "Node: $(hostname)  |  Date: $(date)"
echo "GPU visible: $CUDA_VISIBLE_DEVICES"
$PY -c "import torch; print('cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
echo "═══════════════════════════"
echo ""

$PY run_1s.py

echo ""
echo "════════ JOB DONE  $(date) ════════"
echo "DONE $(date)"
} > $LOG 2>&1
