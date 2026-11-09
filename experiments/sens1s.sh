#!/bin/bash

PY=${PY:-python}
PROJ=${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}
export CUDA_MPS_PIPE_DIRECTORY=/dev/null    # BAT BUOC - fix MPS treo tren compute node
export FI_PROVIDER=tcp    # BAT BUOC - tranh "import torch" bi treo khi khoi tao (CXI/Slingshot)
LOG=$PROJ/logs/sens1s_$(date +%Y%m%d_%H%M%S).log
mkdir -p $PROJ/logs
cd $PROJ
export PYTHONUNBUFFERED=1
{
echo "START $(date) node=$(hostname) CUDA_VISIBLE=$CUDA_VISIBLE_DEVICES"
echo "=== GPU smoke test (khong chan job; xem ghi chu tranh chap MPS) ==="
timeout 300 $PY -u -c "import torch;print('cuda?',torch.cuda.is_available());x=torch.randn(2000,2000).cuda();torch.cuda.synchronize();print('gpu OK',torch.cuda.get_device_name(0))" || echo "!! smoke khong pass -- tiep tuc chay thuc te"

echo "=== Kiem tra do nhay 1s: freq_domain_cnn floor (offset=0) vs round (offset=0.5) ==="
FREQ_BIN_OFFSET=0.0 $PY -u training/train_extractors.py \
    --extractor freq_domain_cnn --data_path data/raw/benchmark --n_subjects 35 \
    --epochs 100 --batch_size 16 --t_start 0.14 --t_end 1.14 \
    --save_dir results/sens_1s_freq_floor
rc=$?
if [ $rc -ne 0 ]; then echo "!! sens floor FAILED (rc=$rc)"; exit 1; fi
FREQ_BIN_OFFSET=0.5 $PY -u training/train_extractors.py \
    --extractor freq_domain_cnn --data_path data/raw/benchmark --n_subjects 35 \
    --epochs 100 --batch_size 16 --t_start 0.14 --t_end 1.14 \
    --save_dir results/sens_1s_freq_round
rc=$?
if [ $rc -ne 0 ]; then echo "!! sens round FAILED (rc=$rc)"; exit 1; fi

echo "DONE $(date)"
} > $LOG 2>&1
