#!/bin/bash

PY=${PY:-python}
PROJ=${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}
export CUDA_MPS_PIPE_DIRECTORY=/dev/null    # BAT BUOC - fix MPS treo tren compute node
export FI_PROVIDER=tcp    # BAT BUOC - tranh "import torch" bi treo 55-185s cho dich vu dam phan CXI (Slingshot) tren node ai01
LOG=$PROJ/logs/ablation_distill_4s_$(date +%Y%m%d_%H%M%S).log
mkdir -p $PROJ/logs
cd $PROJ
export PYTHONUNBUFFERED=1
{
echo "START $(date) node=$(hostname) CUDA_VISIBLE=$CUDA_VISIBLE_DEVICES"
echo "=== GPU smoke test ==="
SMOKE_OK=0
SMOKE_TIMEOUTS=(300 90 90)   # dot 1: du 300s cho "cold-start" dau job (import torch co the mat toi 185s
                             # do dam phan driver mang CXI/Slingshot tren node); dot 2-3: fallback ngan hon
for i in 0 1 2; do
  attempt=$((i+1))
  timeout ${SMOKE_TIMEOUTS[$i]} $PY -u -c "import torch;print('cuda?',torch.cuda.is_available());x=torch.randn(2000,2000).cuda();torch.cuda.synchronize();print('gpu OK',torch.cuda.get_device_name(0))"
  rc=$?
  if [ $rc -eq 0 ]; then SMOKE_OK=1; break; fi
  echo "!! GPU smoke attempt $attempt FAILED (rc=$rc, timeout=${SMOKE_TIMEOUTS[$i]}s) -- known transient CUDA cold-start/CXI hang, retrying in 15s..."
  sleep 15
done
if [ $SMOKE_OK -ne 1 ]; then
  echo "!! GPU smoke test khong pass sau 3 lan thu -- co the do tranh chap MPS voi job khac tren node"
  echo "   dung chung (sharedgpu), KHONG phai GPU hong. Van tiep tuc chay lenh training thuc te"
  echo "   (co ngan sach walltime day du de cho qua tranh chap, khong bi gioi han timeout nhu smoke test)."
fi

echo "=== Ablation co che collaborative distillation @4.0s (cau hinh chinh moi), LOSO 35 fold ==="
echo "--- distill_mode=pairwise (DML goc, tong KL rieng tung cap) ---"
$PY -u training/collaborative_ensemble.py \
    --data_path data/raw/benchmark --init_checkpoint_dir results/baselines_4s_fixedwindow_round \
    --n_subjects 35 --epochs 50 --batch_size 16 \
    --alpha 0.5 --temperature 4.0 \
    --distill_mode pairwise \
    --t_start 0.14 --t_end 4.14 \
    --save_dir results/ablation_distill_pairwise_4s_fixedwindow_round
rc=$?
if [ $rc -ne 0 ]; then echo "!! pairwise (DML) FAILED (rc=$rc)"; exit 1; fi

echo "DONE $(date)"
} > $LOG 2>&1
