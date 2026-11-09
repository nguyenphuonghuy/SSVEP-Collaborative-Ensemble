"""
evaluation/branch_diversity_benchmark.py
==========================================
Branch-diversity analysis:
doi chung CE-only (tabtrainfairness) da tach duoc "them 50 epoch" khoi "co KD",
nhung chua tach duoc "trao doi tri thuc LIEN MIEN" khoi "hieu ung dieu chuan
thuan tuy (label-smoothing-like)" cua soft target. Hai manh bang chung con
thieu, ca hai deu TINH DUOC TU CHECKPOINT DA CO SAN (khong train lai):
  1. Do chinh xac TUNG NHANH truoc/sau fine-tune cong tac.
  2. Chi so da dang (agreement, complementary rate, phi correlation) giua
     tung cap nhanh, truoc/sau fine-tune cong tac.

Khong co training moi - chi forward pass tren checkpoint da luu:
  - TRUOC fine-tune (baseline stage): results/baselines_full_fixedwindow_round/
    {ext}_fold{N}.pt
  - SAU fine-tune (collaborative stage): results/ensemble_full_fixedwindow_round/
    {ext}_collab_fold{N}.pt

Usage:
    python evaluation/branch_diversity_benchmark.py --out results/branch_diversity_fixedwindow_round.json
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.extractors.extractors import build_extractor
from data.benchmark_loader import BenchmarkDataset

EXTRACTORS = ['temporal_cnn', 'freq_domain_cnn', 'spatial_temporal_cnn']


def load_ckpt(ckpt_path, ext_name, n_ch, n_samp, n_cls, device):
    model = build_extractor(ext_name, n_ch, n_samp, n_cls)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def predict(model, X, device, batch_size=64):
    preds = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def pairwise_diversity(preds_by_ext, labels):
    """Agreement / complementary rate / phi correlation cho moi cap nhanh,
    gop tren toan bo mau test cua tat ca fold (giong evaluation/mi_branch_diversity.py)."""
    correct = {e: (preds_by_ext[e] == labels) for e in preds_by_ext}
    out = {}
    names = list(preds_by_ext.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            e1, e2 = names[i], names[j]
            c1, c2 = correct[e1], correct[e2]
            agree = (preds_by_ext[e1] == preds_by_ext[e2]).mean()
            both_correct = (c1 & c2).mean()
            both_wrong = (~c1 & ~c2).mean()
            only1 = (c1 & ~c2).mean()
            only2 = (~c1 & c2).mean()
            complementary = only1 + only2
            phi, _ = pearsonr(c1.astype(float), c2.astype(float))
            out[f"{e1}__vs__{e2}"] = {
                'prediction_agreement_rate': float(agree),
                'both_correct_rate': float(both_correct),
                'both_wrong_rate': float(both_wrong),
                f'only_{e1}_correct_rate': float(only1),
                f'only_{e2}_correct_rate': float(only2),
                'complementary_rate': float(complementary),
                'phi_correctness_correlation': float(phi),
            }
    return out


def run_stage(stage_name, ckpt_dir, ckpt_suffix, X, y, subject_ids, n_ch, n_samp, n_cls, device):
    """stage_name: 'pre_finetune' hoac 'post_finetune'.
    ckpt_suffix: '' -> {ext}_fold{N}.pt (baseline); '_collab' -> {ext}_collab_fold{N}.pt."""
    unique_subjects = np.unique(subject_ids)
    per_fold = []
    preds_by_ext_all = {e: [] for e in EXTRACTORS}
    labels_all = []

    for fold_idx, test_subj in enumerate(unique_subjects):
        fold_num = fold_idx + 1
        test_mask = subject_ids == test_subj
        X_test = torch.FloatTensor(X[test_mask])
        y_test = y[test_mask]

        fold_preds = {}
        missing = False
        for ext in EXTRACTORS:
            ckpt_path = ckpt_dir / f"{ext}{ckpt_suffix}_fold{fold_num}.pt"
            if not ckpt_path.exists():
                print(f"  !! thieu checkpoint {ckpt_path} - bo qua fold {fold_num}")
                missing = True
                break
            model = load_ckpt(ckpt_path, ext, n_ch, n_samp, n_cls, device)
            fold_preds[ext] = predict(model, X_test, device)
        if missing:
            continue

        fold_acc = {ext: float((fold_preds[ext] == y_test).mean()) for ext in EXTRACTORS}
        per_fold.append({'subject': int(test_subj), 'fold': fold_num, 'per_extractor_acc': fold_acc})

        for ext in EXTRACTORS:
            preds_by_ext_all[ext].append(fold_preds[ext])
        labels_all.append(y_test)

        if fold_num % 10 == 0 or fold_num == 1:
            print(f"  [{stage_name}] fold {fold_num}/{len(unique_subjects)} done "
                  f"({', '.join(f'{e}={a:.3f}' for e, a in fold_acc.items())})")

    labels = np.concatenate(labels_all)
    preds_by_ext = {e: np.concatenate(v) for e, v in preds_by_ext_all.items()}

    per_ext_mean_acc = {ext: float(np.mean([f['per_extractor_acc'][ext] for f in per_fold])) for ext in EXTRACTORS}
    per_ext_std_acc = {ext: float(np.std([f['per_extractor_acc'][ext] for f in per_fold])) for ext in EXTRACTORS}
    diversity = pairwise_diversity(preds_by_ext, labels)

    return {
        'stage': stage_name,
        'n_folds': len(per_fold),
        'per_extractor_mean_acc': per_ext_mean_acc,
        'per_extractor_std_acc': per_ext_std_acc,
        'pairwise_diversity_pooled': diversity,
        'per_fold': per_fold,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/raw/benchmark')
    parser.add_argument('--baseline_ckpt_dir', type=str, default='results/baselines_full_fixedwindow_round')
    parser.add_argument('--collab_ckpt_dir', type=str, default='results/ensemble_full_fixedwindow_round')
    parser.add_argument('--n_subjects', type=int, default=35)
    parser.add_argument('--t_start', type=float, default=0.14)
    parser.add_argument('--t_end', type=float, default=5.14)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--out', type=str, default='results/branch_diversity_fixedwindow_round.json')
    args = parser.parse_args()

    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else
                           ('cpu' if args.device == 'auto' else args.device))
    print(f"Device: {device}")

    print(f"Loading Benchmark tu {args.data_path} (t_start={args.t_start}, t_end={args.t_end}) ...")
    loader = BenchmarkDataset(data_path=args.data_path, t_start=args.t_start, t_end=args.t_end, verbose=False)
    X, y, subject_ids = loader.load_all_subjects(n_subjects=args.n_subjects)
    n_ch, n_samp = X.shape[1], X.shape[2]
    n_cls = len(np.unique(y))
    print(f"X={X.shape}, n_classes={n_cls}, n_subjects={len(np.unique(subject_ids))}")

    print("\n=== GIAI DOAN TRUOC FINE-TUNE (baseline stage checkpoints) ===")
    pre = run_stage('pre_finetune', Path(args.baseline_ckpt_dir), '', X, y, subject_ids, n_ch, n_samp, n_cls, device)

    print("\n=== GIAI DOAN SAU FINE-TUNE (collaborative stage checkpoints) ===")
    post = run_stage('post_finetune', Path(args.collab_ckpt_dir), '_collab', X, y, subject_ids, n_ch, n_samp, n_cls, device)

    print("\n" + "=" * 78)
    print("TOM TAT: Do chinh xac tung nhanh TRUOC vs SAU fine-tune cong tac")
    print("=" * 78)
    for ext in EXTRACTORS:
        a0 = pre['per_extractor_mean_acc'][ext]
        a1 = post['per_extractor_mean_acc'][ext]
        print(f"  {ext:24s}: {a0*100:6.2f}% -> {a1*100:6.2f}%  (delta {100*(a1-a0):+.2f}pp)")

    print("\n" + "=" * 78)
    print("TOM TAT: Da dang giua cac cap nhanh TRUOC vs SAU fine-tune cong tac")
    print("=" * 78)
    for pair in pre['pairwise_diversity_pooled']:
        d0 = pre['pairwise_diversity_pooled'][pair]
        d1 = post['pairwise_diversity_pooled'][pair]
        print(f"\n[{pair}]")
        print(f"  agreement rate      : {d0['prediction_agreement_rate']:.4f} -> {d1['prediction_agreement_rate']:.4f}")
        print(f"  complementary rate  : {d0['complementary_rate']:.4f} -> {d1['complementary_rate']:.4f}")
        print(f"  phi (dung/sai corr) : {d0['phi_correctness_correlation']:.4f} -> {d1['phi_correctness_correlation']:.4f}")

    result = {
        'config': {
            'data_path': args.data_path, 't_start': args.t_start, 't_end': args.t_end,
            'n_subjects': args.n_subjects, 'n_samples': n_samp, 'n_channels': n_ch, 'n_classes': n_cls,
            'baseline_ckpt_dir': args.baseline_ckpt_dir, 'collab_ckpt_dir': args.collab_ckpt_dir,
            'timestamp': datetime.now().isoformat(),
        },
        'pre_finetune': pre,
        'post_finetune': post,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
