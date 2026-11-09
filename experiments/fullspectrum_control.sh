#!/bin/bash

PY=${PY:-python}
PROJ=${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}
export CUDA_MPS_PIPE_DIRECTORY=/dev/null
export FI_PROVIDER=tcp
LOG=$PROJ/logs/fullspectrum_fullspectrum_$(date +%Y%m%d_%H%M%S).log
mkdir -p $PROJ/logs
cd $PROJ
export PYTHONUNBUFFERED=1
{
echo "START $(date) node=$(hostname) CUDA_VISIBLE=$CUDA_VISIBLE_DEVICES"
echo "=== GPU smoke test ==="
SMOKE_OK=0
SMOKE_TIMEOUTS=(300 90 90)
for i in 0 1 2; do
  attempt=$((i+1))
  timeout ${SMOKE_TIMEOUTS[$i]} $PY -u -c "import torch;print('cuda?',torch.cuda.is_available());x=torch.randn(2000,2000).cuda();torch.cuda.synchronize();print('gpu OK',torch.cuda.get_device_name(0))"
  rc=$?
  if [ $rc -eq 0 ]; then SMOKE_OK=1; break; fi
  echo "!! GPU smoke attempt $attempt FAILED (rc=$rc), retrying in 15s..."
  sleep 15
done
if [ $SMOKE_OK -ne 1 ]; then
  echo "!! GPU smoke khong pass sau 3 lan - co the tranh chap MPS, van tiep tuc chay thuc te."
fi

echo "=== full-spectrum: doi chung CNN-tren-toan-pho (khong chon bin hai, 6-90Hz, cung tien xu ly) ==="
$PY -u training/train_extractors.py \
    --extractor full_spectrum_cnn \
    --data_path data/raw/benchmark \
    --n_subjects 35 \
    --epochs 100 \
    --batch_size 16 \
    --t_start 0.14 \
    --t_end 5.14 \
    --save_dir results/baselines_fullspectrum_fixedwindow_round \
    --device auto
rc=$?
if [ $rc -ne 0 ]; then echo "!! FAILED (rc=$rc)"; exit 1; fi

echo "DONE $(date)"
} > $LOG 2>&1
