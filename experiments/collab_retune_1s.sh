#!/bin/bash

PY=${PY:-python}
PROJ=${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}
export CUDA_MPS_PIPE_DIRECTORY=/dev/null
export FI_PROVIDER=tcp
LOG=$PROJ/logs/collab_retune_1s_$(date +%Y%m%d_%H%M%S).log
mkdir -p $PROJ/logs
cd $PROJ
export PYTHONUNBUFFERED=1
{
echo "START $(date) node=$(hostname) CUDA_VISIBLE=$CUDA_VISIBLE_DEVICES"
echo "=== GPU smoke test (khong chan job) ==="
timeout 300 $PY -u -c "import torch;print('cuda?',torch.cuda.is_available());x=torch.randn(2000,2000).cuda();torch.cuda.synchronize();print('gpu OK',torch.cuda.get_device_name(0))" || echo "!! smoke khong pass -- tiep tuc chay thuc te"

echo "=== Benchmark @1s, collaborative FULL 35-fold LOSO, alpha=0.7 T=6.0 (chon theo validation tu Run C) ==="
$PY -u training/collaborative_ensemble.py \
    --data_path data/raw/benchmark --init_checkpoint_dir results/baselines_1s_fixedwindow_round \
    --n_subjects 35 --epochs 50 --batch_size 16 \
    --alpha 0.7 --temperature 6.0 \
    --t_start 0.14 --t_end 1.14 \
    --save_dir results/ensemble_1s_fixedwindow_round_hp
rc=$?
if [ $rc -ne 0 ]; then echo "!! collab retune 1s FAILED (rc=$rc)"; exit 1; fi

echo "DONE $(date)"
} > $LOG 2>&1
