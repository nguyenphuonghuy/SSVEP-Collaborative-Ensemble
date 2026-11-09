"""
evaluation/ablation_study.py
==============================
Task 5: Ablation Study — Định lượng đóng góp của từng thành phần.

Các ablation được chạy:
  A. Extractor Ablation  : 4 vs 3 extractors (bỏ lần lượt từng cái)
  B. Method Ablation     : Single best vs Naive vs Collaborative
  C. Hyperparameter      : alpha ∈ {0.3, 0.5, 0.7}, temperature ∈ {2, 4, 6}
  D. Window Ablation     : 0.5s vs 4s (dùng kết quả đã có)

Mỗi ablation dùng lại checkpoints đã train → không cần retrain,
chỉ thay đổi cách combine predictions.

Usage:
    python evaluation/ablation_study.py \
        --checkpoint_dir results/baselines_4s \
        --collab_dir results/ensemble_4s \
        --data_path data/raw/benchmark \
        --t_start 0.14 --t_end 4.14
"""

import sys
import json
import logging
import argparse
import itertools
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.extractors.extractors import build_extractor
from data.benchmark_loader import BenchmarkDataset
from data.beta_loader import BETADataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

ALL_EXTRACTORS = [
    'temporal_cnn',
    'freq_domain_cnn',
    'shallow_conv_net',
    'spatial_temporal_cnn',
]

# Kien truc cuoi cung (sau khi bo ShallowConvNet) - dung cho Ablation C,
# vi Bang 6 khao sat do nhay alpha/T cua chinh mo hinh 3 nhanh duoc dung
# xuyen suot cac thi nghiem, khong phai cau hinh 4 nhanh ban dau.
FINAL_EXTRACTORS = [
    'temporal_cnn',
    'freq_domain_cnn',
    'spatial_temporal_cnn',
]

EXT_LABELS = {
    'temporal_cnn':        'Ext1-Temporal',
    'freq_domain_cnn':     'Ext2-FreqDomain',
    'shallow_conv_net':    'Ext3-Shallow',
    'spatial_temporal_cnn':'Ext4-Spatial',
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_model(name, fold, ckpt_dir, n_ch, n_samp, n_cls, device):
    path = Path(ckpt_dir) / f"{name}_fold{fold}.pt"
    m = build_extractor(name, n_ch, n_samp, n_cls)
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    return m


def load_collab_model(name, fold, collab_dir, n_ch, n_samp, n_cls, device):
    """Load collaborative checkpoint; fallback to baseline nếu không có."""
    path = Path(collab_dir) / f"{name}_collab_fold{fold}.pt"
    if not path.exists():
        logger.warning(f"Collab ckpt không có: {path}, dùng baseline")
        return None
    m = build_extractor(name, n_ch, n_samp, n_cls)
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    return m


@torch.no_grad()
def avg_ensemble_acc(models, X_test, y_test, device, batch_size=64):
    """Average softmax ensemble accuracy."""
    ds = TensorDataset(torch.FloatTensor(X_test))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_probs = []
    for (xb,) in loader:
        xb = xb.to(device)
        probs = torch.stack([F.softmax(m(xb), dim=-1) for m in models]).mean(0)
        all_probs.append(probs.cpu())
    probs = torch.cat(all_probs).numpy()
    return (probs.argmax(1) == y_test).mean()


@torch.no_grad()
def single_acc(model, X_test, y_test, device, batch_size=64):
    ds = TensorDataset(torch.FloatTensor(X_test))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    correct, total = 0, 0
    for (xb,) in loader:
        xb = xb.to(device)
        pred = model(xb).argmax(1).cpu().numpy()
        correct += (pred == y_test[:len(pred)]).sum()
        y_test = y_test[len(pred):]
        total += len(pred)
    return correct / total


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION A: Extractor Subset
# ─────────────────────────────────────────────────────────────────────────────

def ablation_extractor_subset(
    X, y, subject_ids, ckpt_dir, device, n_ch, n_samp, n_cls, batch_size
):
    """
    So sánh:
    - Full 4 extractors
    - Bỏ lần lượt từng extractor (4 variants, mỗi cái 3 extractors)
    - Best single extractor
    """
    logger.info("\n=== Ablation A: Extractor Subset ===")
    unique_subjects = np.unique(subject_ids)
    results = {}

    # Tất cả configs cần test
    configs = {
        'all_4': ALL_EXTRACTORS,
        'drop_ext1': [e for e in ALL_EXTRACTORS if e != 'temporal_cnn'],
        'drop_ext2': [e for e in ALL_EXTRACTORS if e != 'freq_domain_cnn'],
        'drop_ext3': [e for e in ALL_EXTRACTORS if e != 'shallow_conv_net'],
        'drop_ext4': [e for e in ALL_EXTRACTORS if e != 'spatial_temporal_cnn'],
        'best_single_ext1': ['temporal_cnn'],
        'best_single_ext2': ['freq_domain_cnn'],
    }

    for config_name, ext_list in configs.items():
        fold_accs = []
        val_accs = []
        for fold_idx, test_subj in enumerate(unique_subjects):
            fold_num = fold_idx + 1
            test_mask = subject_ids == test_subj
            X_test = X[test_mask]
            y_test = y[test_mask]

            # Tai tao LAI dung tap validation da dung de chon checkpoint trong
            # train_extractors.py (cung seed 1000+fold_idx) - dung tap nay de
            # QUYET DINH kien truc, khong dung test subject. Test subject chi
            # dung de bao cao (informational), khong anh huong quyet dinh nao.
            train_pool = unique_subjects[unique_subjects != test_subj]
            rng = np.random.RandomState(1000 + fold_idx)
            shuffled = rng.permutation(train_pool)
            n_val = min(max(3, round(0.15 * len(train_pool))), max(1, len(train_pool) - 1))
            val_subjects = set(shuffled[:n_val].tolist())
            val_mask = np.isin(subject_ids, list(val_subjects))
            X_val, y_val = X[val_mask], y[val_mask]

            models = []
            ok = True
            for ext in ext_list:
                try:
                    m = load_model(ext, fold_num, ckpt_dir,
                                   n_ch, n_samp, n_cls, device)
                    models.append(m)
                except FileNotFoundError:
                    ok = False
                    break

            if not ok:
                continue

            if len(models) == 1:
                acc = single_acc(models[0], X_test, y_test, device, batch_size)
                vacc = single_acc(models[0], X_val, y_val, device, batch_size)
            else:
                acc = avg_ensemble_acc(models, X_test, y_test, device, batch_size)
                vacc = avg_ensemble_acc(models, X_val, y_val, device, batch_size)
            fold_accs.append(acc)
            val_accs.append(vacc)

        mean_acc = np.mean(fold_accs) if fold_accs else 0.0
        mean_val_acc = np.mean(val_accs) if val_accs else 0.0
        results[config_name] = {
            'extractors': ext_list,
            'mean_acc': float(mean_acc),
            'std_acc': float(np.std(fold_accs)) if fold_accs else 0.0,
            'mean_val_acc': float(mean_val_acc),
            'std_val_acc': float(np.std(val_accs)) if val_accs else 0.0,
            'n_folds': len(fold_accs),
        }
        logger.info(f"  {config_name:<20}: test={mean_acc:.4f}±{np.std(fold_accs):.4f} "
                    f"| val={mean_val_acc:.4f}±{np.std(val_accs):.4f}"
                    f"  ({', '.join([EXT_LABELS.get(e,e) for e in ext_list])})")

    # Quyet dinh kien truc DUA TREN VALIDATION (khong dua tren test) - xem
    # Muc 3.4/5.3.1 trong bai bao.
    best_by_val = max(
        (k for k in results if k.startswith('drop_') or k == 'all_4'),
        key=lambda k: results[k]['mean_val_acc']
    )
    logger.info(f"\n  Cau hinh tot nhat theo VALIDATION: {best_by_val} "
                f"(val={results[best_by_val]['mean_val_acc']:.4f}, "
                f"test={results[best_by_val]['mean_acc']:.4f})")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION B: Method Comparison
# ─────────────────────────────────────────────────────────────────────────────

def ablation_method_comparison(
    X, y, subject_ids, ckpt_dir, collab_dir, device, n_ch, n_samp, n_cls, batch_size
):
    """
    So sánh các phương pháp:
    - Single best extractor (Ext1)
    - Naive ensemble (avg softmax, no training)
    - Collaborative ensemble (fine-tuned)
    """
    logger.info("\n=== Ablation B: Method Comparison ===")
    unique_subjects = np.unique(subject_ids)

    single_accs, naive_accs, collab_accs = [], [], []

    for fold_idx, test_subj in enumerate(unique_subjects):
        fold_num = fold_idx + 1
        test_mask = subject_ids == test_subj
        X_test = X[test_mask]
        y_test = y[test_mask]

        # Load baseline models
        base_models = []
        ok = True
        for ext in ALL_EXTRACTORS:
            try:
                m = load_model(ext, fold_num, ckpt_dir,
                               n_ch, n_samp, n_cls, device)
                base_models.append(m)
            except FileNotFoundError:
                ok = False
                break
        if not ok:
            continue

        # Single best (Ext1 TemporalCNN)
        s_acc = single_acc(base_models[0], X_test, y_test, device, batch_size)
        single_accs.append(s_acc)

        # Naive ensemble
        n_acc = avg_ensemble_acc(base_models, X_test, y_test, device, batch_size)
        naive_accs.append(n_acc)

        # Collaborative ensemble
        collab_models = []
        collab_ok = True
        for ext in ALL_EXTRACTORS:
            m = load_collab_model(ext, fold_num, collab_dir,
                                  n_ch, n_samp, n_cls, device)
            if m is None:
                collab_ok = False
                break
            collab_models.append(m)

        if collab_ok:
            c_acc = avg_ensemble_acc(collab_models, X_test, y_test, device, batch_size)
            collab_accs.append(c_acc)

    results = {
        'single_best': {
            'extractor': 'temporal_cnn',
            'mean_acc': float(np.mean(single_accs)),
            'std_acc': float(np.std(single_accs)),
        },
        'naive_ensemble': {
            'mean_acc': float(np.mean(naive_accs)),
            'std_acc': float(np.std(naive_accs)),
        },
        'collaborative': {
            'mean_acc': float(np.mean(collab_accs)) if collab_accs else None,
            'std_acc': float(np.std(collab_accs)) if collab_accs else None,
        },
    }

    logger.info(f"  Single best (Ext1)  : {results['single_best']['mean_acc']:.4f}"
                f" ± {results['single_best']['std_acc']:.4f}")
    logger.info(f"  Naive ensemble      : {results['naive_ensemble']['mean_acc']:.4f}"
                f" ± {results['naive_ensemble']['std_acc']:.4f}")
    if results['collaborative']['mean_acc']:
        logger.info(f"  Collaborative       : {results['collaborative']['mean_acc']:.4f}"
                    f" ± {results['collaborative']['std_acc']:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION C: Hyperparameter Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def train_collab_one_fold(
    models_init_ckpts, ext_names, X_train, y_train, X_val, y_val, X_test, y_test,
    alpha, temperature, epochs, lr, batch_size, device, n_ch, n_samp, n_cls
):
    """Train collaborative cho 1 fold. Chon epoch theo VALIDATION; test chi danh gia 1 lan."""
    import copy
    models = []
    for name, ckpt in zip(ext_names, models_init_ckpts):
        m = build_extractor(name, n_ch, n_samp, n_cls)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.to(device)
        models.append(m)

    ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)

    optimizers = [optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
                  for m in models]

    ce_fn = nn.CrossEntropyLoss()
    T = temperature

    def ensemble_acc(X_eval, y_eval):
        for m in models:
            m.eval()
        with torch.no_grad():
            Xt = torch.FloatTensor(X_eval).to(device)
            probs = torch.stack([F.softmax(m(Xt), dim=-1)
                                 for m in models]).mean(0)
            return float((probs.argmax(1).cpu().numpy() == y_eval).mean())

    best_val, best_states = -1.0, None
    for epoch in range(epochs):
        for m in models:
            m.train()

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            all_logits = [m(xb) for m in models]
            all_soft = [F.softmax(lg / T, dim=-1).detach() for lg in all_logits]

            for i, (model, opt, logits) in enumerate(
                zip(models, optimizers, all_logits)
            ):
                others = [all_soft[j] for j in range(len(models)) if j != i]
                soft_target = torch.stack(others).mean(0)

                opt.zero_grad(set_to_none=True)
                loss_ce = ce_fn(logits, yb)
                log_pi = F.log_softmax(logits / T, dim=-1)
                loss_kd = F.kl_div(log_pi, soft_target,
                                   reduction='batchmean') * (T ** 2)
                loss = alpha * loss_ce + (1 - alpha) * loss_kd
                loss.backward(retain_graph=(i < len(models) - 1))
                opt.step()

        vacc = ensemble_acc(X_val, y_val)
        if vacc > best_val:
            best_val = vacc
            best_states = [copy.deepcopy(m.state_dict()) for m in models]

    for m, st in zip(models, best_states):
        m.load_state_dict(st)
    test_acc = ensemble_acc(X_test, y_test)
    return best_val, test_acc


def ablation_hyperparams(
    X, y, subject_ids, ckpt_dir, device, n_ch, n_samp, n_cls,
    batch_size, epochs=30, n_folds_sample=10
):
    """
    Grid search nhỏ trên alpha × temperature.
    Chỉ chạy n_folds_sample folds đầu để tiết kiệm thời gian.

    alpha       ∈ {0.3, 0.5, 0.7}
    temperature ∈ {2.0, 4.0, 6.0}
    """
    logger.info(f"\n=== Ablation C: Hyperparameter Sensitivity "
                f"({n_folds_sample} folds sample) ===")

    alphas = [0.3, 0.5, 0.7]
    temperatures = [2.0, 4.0, 6.0]

    unique_subjects_all = np.unique(subject_ids)
    unique_subjects = unique_subjects_all[:n_folds_sample]
    results = {}

    for alpha, temp in itertools.product(alphas, temperatures):
        key = f"alpha={alpha}_T={temp}"
        fold_accs = []
        fold_vaccs = []

        for fold_idx, test_subj in enumerate(unique_subjects):
            fold_num = fold_idx + 1
            test_mask  = subject_ids == test_subj
            train_mask = ~(subject_ids == test_subj)

            # Tach validation GIONG HET pipeline chinh (cung RNG, cung ty le)
            train_pool = unique_subjects_all[unique_subjects_all != test_subj]
            rng = np.random.RandomState(1000 + fold_idx)
            shuffled = rng.permutation(train_pool)
            n_val = min(max(3, round(0.15 * len(train_pool))), max(1, len(train_pool) - 1))
            val_subjects = set(shuffled[:n_val].tolist())
            val_mask   = np.isin(subject_ids, list(val_subjects))
            train_mask = np.isin(subject_ids, list(set(shuffled[n_val:].tolist())))

            X_train, y_train = X[train_mask], y[train_mask]
            X_val,   y_val   = X[val_mask],   y[val_mask]
            X_test,  y_test  = X[test_mask],  y[test_mask]

            # Collect checkpoint paths (kien truc cuoi cung, 3 nhanh)
            ckpts = [Path(ckpt_dir) / f"{ext}_fold{fold_num}.pt"
                     for ext in FINAL_EXTRACTORS]
            if not all(c.exists() for c in ckpts):
                continue

            vacc, acc = train_collab_one_fold(
                models_init_ckpts=ckpts,
                ext_names=FINAL_EXTRACTORS,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test,
                alpha=alpha, temperature=temp,
                epochs=epochs, lr=0.0003,
                batch_size=batch_size, device=device,
                n_ch=n_ch, n_samp=n_samp, n_cls=n_cls
            )
            fold_accs.append(acc)
            fold_vaccs.append(vacc)

        if not fold_accs:
            logger.warning(f"  !! {key}: khong co fold nao chay duoc "
                           f"(thieu checkpoint trong {ckpt_dir}) - BO QUA config nay")
            continue
        mean_acc = float(np.mean(fold_accs))
        results[key] = {
            'alpha': alpha,
            'temperature': temp,
            'mean_val_acc': float(np.mean(fold_vaccs)),
            'mean_acc': mean_acc,
            'std_acc': float(np.std(fold_accs)) if fold_accs else 0.0,
            'n_folds': len(fold_accs),
        }
        logger.info(f"  alpha={alpha}, T={temp}: {mean_acc:.4f}"
                    f" ± {np.std(fold_accs):.4f}")

    # Best config
    best_key = max(results, key=lambda k: results[k]['mean_val_acc'])
    logger.info(f"\n  Best config (chon theo VAL): {best_key} → test={results[best_key]['mean_acc']:.4f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Task 5: Ablation Study')
    parser.add_argument('--data_path', type=str, default='data/raw/benchmark')
    parser.add_argument('--dataset', type=str, default='benchmark',
                        choices=['benchmark', 'beta'],
                        help='Dataset de danh gia: benchmark hoac beta '
                             '(dung "beta" cho Ablation C de tach biet hoan '
                             'toan voi 35 subject cua Benchmark dung o ket qua chinh)')
    parser.add_argument('--checkpoint_dir', type=str, default='results/baselines_4s')
    parser.add_argument('--collab_dir', type=str, default='results/ensemble_4s')
    parser.add_argument('--save_dir', type=str, default='results/ablation')
    parser.add_argument('--n_subjects', type=int, default=35)
    parser.add_argument('--t_start', type=float, default=0.14)
    parser.add_argument('--t_end', type=float, default=4.14)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='auto')
    # Ablation C params
    parser.add_argument('--hp_epochs', type=int, default=30,
                        help='Epochs cho hyperparameter ablation (ít hơn để nhanh)')
    parser.add_argument('--hp_n_folds', type=int, default=10,
                        help='Số folds sample cho hyperparameter ablation')
    # Chọn ablation nào chạy
    parser.add_argument('--run', type=str, default='all',
                        choices=['all', 'A', 'B', 'C'],
                        help='Ablation nào muốn chạy: all / A / B / C')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Load data
    logger.info(f"Loading {args.dataset} dataset từ {args.data_path} ...")
    if args.dataset == 'beta':
        loader = BETADataset(
            data_path=args.data_path,
            t_start=args.t_start,
            t_end=args.t_end,
            verbose=False
        )
    else:
        loader = BenchmarkDataset(
            data_path=args.data_path,
            t_start=args.t_start,
            t_end=args.t_end,
            verbose=False
        )
    X, y, subject_ids = loader.load_all_subjects(n_subjects=args.n_subjects)
    n_ch, n_samp, n_cls = X.shape[1], X.shape[2], len(np.unique(y))
    logger.info(f"Data: X={X.shape}, classes={n_cls}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results = {'config': vars(args)}

    # ── Ablation A ──
    if args.run in ('all', 'A'):
        res_a = ablation_extractor_subset(
            X, y, subject_ids,
            ckpt_dir=args.checkpoint_dir,
            device=device, n_ch=n_ch, n_samp=n_samp, n_cls=n_cls,
            batch_size=args.batch_size
        )
        all_results['ablation_A_extractor_subset'] = res_a

    # ── Ablation B ──
    if args.run in ('all', 'B'):
        res_b = ablation_method_comparison(
            X, y, subject_ids,
            ckpt_dir=args.checkpoint_dir,
            collab_dir=args.collab_dir,
            device=device, n_ch=n_ch, n_samp=n_samp, n_cls=n_cls,
            batch_size=args.batch_size
        )
        all_results['ablation_B_method_comparison'] = res_b

    # ── Ablation C ──
    if args.run in ('all', 'C'):
        res_c = ablation_hyperparams(
            X, y, subject_ids,
            ckpt_dir=args.checkpoint_dir,
            device=device, n_ch=n_ch, n_samp=n_samp, n_cls=n_cls,
            batch_size=args.batch_size,
            epochs=args.hp_epochs,
            n_folds_sample=args.hp_n_folds
        )
        all_results['ablation_C_hyperparams'] = res_c

    # ── Save ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    out_path = save_dir / f"ablation_results_{timestamp}.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nAll results saved: {out_path}")

    # ── Print summary ──
    print("\n" + "="*70)
    print("ABLATION STUDY — SUMMARY")
    print("="*70)

    if 'ablation_A_extractor_subset' in all_results:
        print("\n[A] Extractor Subset:")
        for k, v in all_results['ablation_A_extractor_subset'].items():
            print(f"  {k:<20}: {v['mean_acc']:.4f} ± {v['std_acc']:.4f}")

    if 'ablation_B_method_comparison' in all_results:
        print("\n[B] Method Comparison:")
        b = all_results['ablation_B_method_comparison']
        print(f"  Single best         : {b['single_best']['mean_acc']:.4f}")
        print(f"  Naive ensemble      : {b['naive_ensemble']['mean_acc']:.4f}")
        if b['collaborative']['mean_acc']:
            print(f"  Collaborative       : {b['collaborative']['mean_acc']:.4f}")

    if 'ablation_C_hyperparams' in all_results:
        print("\n[C] Hyperparameter Sensitivity:")
        c = all_results['ablation_C_hyperparams']
        # Chon theo VALIDATION (khop voi lua chon thuc te trong ablation_hyperparams),
        # KHONG theo mean_acc (test) -- de khong hien thi mot "best" khac voi lua
        # chon khong-ro-ri-test da dung.
        best = max(c, key=lambda k: c[k]['mean_val_acc'])
        print(f"  Best config (chon theo VAL): {best} → test={c[best]['mean_acc']:.4f}")
        for k, v in sorted(c.items(), key=lambda x: -x[1]['mean_val_acc'])[:5]:
            print(f"  {k:<25}: test={v['mean_acc']:.4f} (val={v['mean_val_acc']:.4f})")

    print("="*70)


if __name__ == "__main__":
    main()
