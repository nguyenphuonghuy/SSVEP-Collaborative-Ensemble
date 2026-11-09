#!/bin/bash

PY=${PY:-python}
PROJ=${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}
export CUDA_MPS_PIPE_DIRECTORY=/dev/null
export FI_PROVIDER=tcp
LOG=$PROJ/logs/multiseed_seed2002_$(date +%Y%m%d_%H%M%S).log
mkdir -p $PROJ/logs
cd $PROJ
export PYTHONUNBUFFERED=1
SEED=2002
BASE_DIR=results/baselines_full_fixedwindow_round_seed${SEED}
ENS_DIR=results/ensemble_full_fixedwindow_round_seed${SEED}
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

echo "=== multi-seed: lap lai GIAI DOAN BASELINE (khong chi fine-tune) voi seed=$SEED doc lap ==="
echo "Khac voi 5-seed variance decomp da co (chi seed giai doan fine-tune), o day seed con"
echo "dieu khien CA khoi tao trong so + thu tu shuffle cua GIAI DOAN BASELINE (100 epoch/fold)."

for ext in temporal_cnn freq_domain_cnn spatial_temporal_cnn; do
  echo "--- [1/3] baseline extractor: $ext (seed=$SEED) ---"
  $PY -u training/train_extractors.py \
      --extractor $ext \
      --data_path data/raw/benchmark --n_subjects 35 \
      --epochs 100 --batch_size 16 \
      --t_start 0.14 --t_end 5.14 \
      --save_dir $BASE_DIR \
      --seed $SEED \
      --device auto
  rc=$?
  if [ $rc -ne 0 ]; then echo "!! Train $ext FAILED (rc=$rc)"; exit 1; fi
done

echo "--- [2/3] naive ensemble (seed=$SEED baseline) ---"
$PY -u evaluation/naive_ensemble.py \
    --data_path data/raw/benchmark \
    --checkpoint_dir $BASE_DIR \
    --n_subjects 35 \
    --t_start 0.14 --t_end 5.14 \
    --save_dir $ENS_DIR
rc=$?
if [ $rc -ne 0 ]; then echo "!! naive_ensemble FAILED (rc=$rc)"; exit 1; fi

echo "--- [3/3] collaborative fine-tune (tren baseline seed=$SEED) ---"
$PY -u training/collaborative_ensemble.py \
    --data_path data/raw/benchmark \
    --init_checkpoint_dir $BASE_DIR \
    --n_subjects 35 \
    --save_dir $ENS_DIR \
    --epochs 50 --batch_size 16 \
    --alpha 0.5 --temperature 4.0 --distill_mode avg \
    --t_start 0.14 --t_end 5.14 \
    --seed $SEED \
    --device auto
rc=$?
if [ $rc -ne 0 ]; then echo "!! collaborative FAILED (rc=$rc)"; exit 1; fi

echo "DONE $(date)"
} > $LOG 2>&1
