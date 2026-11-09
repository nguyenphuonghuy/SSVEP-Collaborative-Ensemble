#!/bin/bash

PY=${PY:-python}
PROJ=${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}
export CUDA_MPS_PIPE_DIRECTORY=/dev/null    # BAT BUOC - fix MPS treo tren compute node
export FI_PROVIDER=tcp    # BAT BUOC - tranh "import torch" bi treo khi khoi tao (CXI/Slingshot)
LOG=$PROJ/logs/round5s_ens_$(date +%Y%m%d_%H%M%S).log
mkdir -p $PROJ/logs
cd $PROJ
export PYTHONUNBUFFERED=1
{
echo "START $(date) node=$(hostname) CUDA_VISIBLE=$CUDA_VISIBLE_DEVICES"
echo "=== GPU smoke test (khong chan job; xem ghi chu tranh chap MPS) ==="
timeout 300 $PY -u -c "import torch;print('cuda?',torch.cuda.is_available());x=torch.randn(2000,2000).cuda();torch.cuda.synchronize();print('gpu OK',torch.cuda.get_device_name(0))" || echo "!! smoke khong pass -- tiep tuc chay thuc te"

echo "=== 5s: naive + collaborative voi freq round (bo qua baseline) ==="
$PY -u run_full.py --skip-baseline
rc=$?
if [ $rc -ne 0 ]; then echo "!! run_full (naive+collab) FAILED (rc=$rc)"; exit 1; fi

echo "DONE $(date)"
} > $LOG 2>&1
