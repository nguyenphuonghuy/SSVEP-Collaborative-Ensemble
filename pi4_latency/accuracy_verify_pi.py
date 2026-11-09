"""
accuracy_verify_pi.py
======================
Xac minh do chinh xac suy luan THUC (dung trong so da huan luyen, khong
phai trong so khoi tao ngau nhien nhu latency_benchmark_pi.py) tren
Raspberry Pi 4 (CPU, aarch64, PyTorch build rieng cho ARM), doi chieu voi
ket qua da tinh tren GPU (H100, CUDA) trong collaborative_loso_*.json.

Muc dich: xac nhan viec suy luan tren mot backend/kien truc CPU khac
(ARM thay vi CUDA) khong lam sai lech quyet dinh phan loai cua ensemble -
cau hoi thuc te khi trien khai mo hinh da huan luyen tren thiet bi bien.

Usage (tren Pi 4, sau khi da dong bo checkpoint + du lieu npz):
    python accuracy_verify_pi.py --subject 33 \
        --ckpt_dir ckpts --data_dir accuracy_verify_data --out acc_s33.json
"""
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from models.extractors.extractors import build_extractor

N_CHANNELS = 9
N_CLASSES = 40
EXTRACTORS = ['temporal_cnn', 'freq_domain_cnn', 'spatial_temporal_cnn']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', type=int, required=True)
    ap.add_argument('--ckpt_dir', type=str, required=True,
                     help="Thu muc chua {ext}_collab_fold{subject}.pt")
    ap.add_argument('--data_dir', type=str, required=True,
                     help="Thu muc chua subject{subject}_test.npz (X,y)")
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()
    args.out = args.out or f'accuracy_verify_s{args.subject}.json'

    data = np.load(Path(args.data_dir) / f'subject{args.subject}_test.npz')
    X = torch.FloatTensor(data['X'])
    y = torch.LongTensor(data['y'])
    n_samples = X.shape[-1]

    models = []
    for name in EXTRACTORS:
        ckpt = Path(args.ckpt_dir) / f'{name}_collab_fold{args.subject}.pt'
        m = build_extractor(name, N_CHANNELS, n_samples, N_CLASSES)
        m.load_state_dict(torch.load(ckpt, map_location='cpu'))
        m.eval()
        models.append(m)

    with torch.no_grad():
        all_probs = []
        per_ext_correct = [0] * len(models)
        for i, m in enumerate(models):
            logits = m(X)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)
            per_ext_correct[i] = int((logits.argmax(1) == y).sum().item())
        avg_prob = torch.stack(all_probs, dim=0).mean(dim=0)
        pred = avg_prob.argmax(1)
        n_correct = int((pred == y).sum().item())

    n_total = len(y)
    result = {
        'device': 'Raspberry Pi 4 (CPU, ARM, trong so da huan luyen tu GPU)',
        'subject': args.subject,
        'n_trials': n_total,
        'ensemble_acc_pi4': n_correct / n_total,
        'per_extractor_acc_pi4': {
            name: c / n_total for name, c in zip(EXTRACTORS, per_ext_correct)
        },
        'predictions': pred.tolist(),
        'labels': y.tolist(),
    }
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Subject {args.subject}: ensemble_acc_pi4={result['ensemble_acc_pi4']*100:.2f}% "
          f"({n_correct}/{n_total} trials) -> {args.out}")


if __name__ == '__main__':
    main()
