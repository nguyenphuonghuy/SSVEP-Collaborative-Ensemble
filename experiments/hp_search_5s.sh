#!/bin/bash

PY=${PY:-python}
PROJ=${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}
export CUDA_MPS_PIPE_DIRECTORY=/dev/null
export FI_PROVIDER=tcp
LOG=$PROJ/logs/hp_search_5s_$(date +%Y%m%d_%H%M%S).log
mkdir -p $PROJ/logs
cd $PROJ
export PYTHONUNBUFFERED=1
{
echo "START $(date) node=$(hostname) CUDA_VISIBLE=$CUDA_VISIBLE_DEVICES"
echo "=== GPU smoke test (khong chan job) ==="
timeout 300 $PY -u -c "import torch;print('cuda?',torch.cuda.is_available());x=torch.randn(2000,2000).cuda();torch.cuda.synchronize();print('gpu OK',torch.cuda.get_device_name(0))" || echo "!! smoke khong pass -- tiep tuc chay thuc te"

echo "=== Run C (val-selected): grid alpha x T, Benchmark @5s, 10-fold sample ==="
$PY -u evaluation/ablation_study.py --run C \
    --dataset benchmark --data_path data/raw/benchmark --n_subjects 35 \
    --checkpoint_dir results/baselines_full_fixedwindow_round \
    --t_start 0.14 --t_end 5.14 --batch_size 16 \
    --hp_epochs 30 --hp_n_folds 10 \
    --save_dir results/ablation_hp_5s_fixedwindow_round
rc=$?
if [ $rc -ne 0 ]; then echo "!! hp search 5s FAILED (rc=$rc)"; exit 1; fi

echo "DONE $(date)"
} > $LOG 2>&1
