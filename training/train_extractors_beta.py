"""
training/train_extractors_beta.py
===================================
Task 6: Train 3-extractor ensemble trên BETA dataset với LOSO.

Giống train_extractors.py nhưng dùng BETADataset loader.
BETA specs: 70 subjects, 4 blocks, 40 targets, 3s trial (750 samples @ 250Hz)
Dùng window 2s (t_start=0.14, t_end=2.14 → 500 samples) để fit trong 3s.

Usage:
    # Test 1 fold nhanh
    python training/train_extractors_beta.py \
        --extractor temporal_cnn \
        --data_path data/raw/beta \
        --n_folds 1 --epochs 20

    # Full LOSO 70 subjects
    python training/train_extractors_beta.py \
        --extractor temporal_cnn \
        --data_path data/raw/beta \
        --epochs 100 --save_dir results/baselines_beta
"""

import sys
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.extractors.extractors import build_extractor, EXTRACTOR_REGISTRY
from data.beta_loader import BETADataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 3 extractors (từ ablation study)
BETA_EXTRACTORS = [
    'temporal_cnn',
    'freq_domain_cnn',
    'spatial_temporal_cnn',
]


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return {'loss': total_loss / total, 'acc': correct / total}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return {'loss': total_loss / total, 'acc': correct / total}


def train_loso_beta(
    extractor_name, X, y, subject_ids, config, device, save_dir, n_folds=None
):
    unique_subjects = np.unique(subject_ids)
    if n_folds:
        unique_subjects = unique_subjects[:n_folds]

    n_ch, n_samp = X.shape[1], X.shape[2]
    n_cls = len(np.unique(y))

    logger.info(f"\n{'='*60}")
    logger.info(f"BETA LOSO: {extractor_name}")
    logger.info(f"  Subjects={len(unique_subjects)}, Classes={n_cls}")
    logger.info(f"  Input=({n_ch}ch × {n_samp}samples)")
    logger.info(f"{'='*60}")

    fold_results = []

    for fold_idx, test_subj in enumerate(unique_subjects):
        fold_num = fold_idx + 1
        fold_start = time.time()

        # Test subject hoan toan tach biet. Tach them validation theo SUBJECT
        # tu cac subject train de chon checkpoint - test subject khong tham
        # gia bat ky quyet dinh nao, chi danh gia 1 lan duy nhat cuoi cung.
        train_pool = unique_subjects[unique_subjects != test_subj]
        rng = np.random.RandomState(1000 + fold_idx)
        shuffled = rng.permutation(train_pool)
        n_val = min(max(3, round(0.15 * len(train_pool))), max(1, len(train_pool) - 1))
        val_subjects = set(shuffled[:n_val].tolist())
        train_subjects = set(shuffled[n_val:].tolist())

        test_mask = subject_ids == test_subj
        val_mask  = np.isin(subject_ids, list(val_subjects))
        train_mask = np.isin(subject_ids, list(train_subjects))

        X_test,  y_test  = X[test_mask],  y[test_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]
        test_ds = TensorDataset(
            torch.FloatTensor(X_test), torch.LongTensor(y_test)
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val), torch.LongTensor(y_val)
        )
        test_loader  = DataLoader(test_ds,  batch_size=config['batch_size'],
                                  shuffle=False, num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'],
                                  shuffle=False, num_workers=2, pin_memory=True)

        ckpt_path = save_dir / f"{extractor_name}_fold{fold_num}.pt"
        done_path = save_dir / f"{extractor_name}_fold{fold_num}.done.json"
        if done_path.exists() and ckpt_path.exists():
            # Resume: checkpoint da co san (vi du chay lai sau khi bi kill giua chung
            # do het walltime) -> danh gia lai thay vi train lai tu dau.
            model = build_extractor(extractor_name, n_ch, n_samp, n_cls).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            resumed_stats = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
            fold_results.append({
                'subject': int(test_subj),
                'fold': fold_num,
                'best_acc': float(resumed_stats['acc']),
                'best_epoch': None,
                'time_sec': 0.0,
            })
            logger.info(f"  Fold {fold_num:2d}/{len(unique_subjects)} | S{test_subj:02d} | "
                        f"da co checkpoint, danh gia lai (resume) | Acc={resumed_stats['acc']:.4f}")
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        train_ds = TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                                  shuffle=True, num_workers=2, pin_memory=True)

        model = build_extractor(extractor_name, n_ch, n_samp, n_cls).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                               weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs']
        )
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

        best_val_acc, best_epoch = 0.0, 0

        for epoch in range(config['epochs']):
            train_stats = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler
            )
            val_stats = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if val_stats['acc'] > best_val_acc:
                best_val_acc = val_stats['acc']
                best_epoch = epoch
                torch.save(
                    model.state_dict(),
                    save_dir / f"{extractor_name}_fold{fold_num}.pt"
                )

            if (epoch + 1) % 20 == 0 or epoch == 0:
                logger.info(
                    f"  Fold {fold_num:2d}/{len(unique_subjects)} | "
                    f"Ep {epoch+1:3d}/{config['epochs']} | "
                    f"Train={train_stats['acc']:.3f} | "
                    f"Val={val_stats['acc']:.3f} | "
                    f"Best(val)={best_val_acc:.3f}"
                )

        # Danh gia tren test DUNG MOT LAN DUY NHAT bang checkpoint da chon
        # boi validation.
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        final_test_stats = evaluate(model, test_loader, criterion, device)
        best_acc = final_test_stats['acc']

        fold_time = time.time() - fold_start
        fold_results.append({
            'subject': int(test_subj),
            'fold': fold_num,
            'best_acc': float(best_acc),
            'best_val_acc': float(best_val_acc),
            'best_epoch': best_epoch,
            'time_sec': fold_time,
        })
        with open(done_path, 'w') as f:
            json.dump(fold_results[-1], f)
        logger.info(
            f"  ✓ Fold {fold_num:2d} | S{test_subj:02d} | "
            f"Val={best_val_acc:.4f} → Test={best_acc:.4f} ({fold_time:.1f}s)"
        )

    accs = [r['best_acc'] for r in fold_results]
    result = {
        'extractor': extractor_name,
        'dataset': 'BETA',
        'mean_acc': float(np.mean(accs)),
        'std_acc':  float(np.std(accs)),
        'min_acc':  float(np.min(accs)),
        'max_acc':  float(np.max(accs)),
        'n_subjects': len(fold_results),
        'folds': fold_results,
        'config': config,
    }

    logger.info(f"\n  BETA LOSO {extractor_name}: "
                f"{result['mean_acc']:.4f} ± {result['std_acc']:.4f}")

    out = save_dir / f"{extractor_name}_beta_loso_results.json"
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"  Saved: {out}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Train extractors on BETA dataset')
    parser.add_argument('--extractor', type=str, default='all',
                        choices=BETA_EXTRACTORS + ['shallow_conv_net', 'freq_domain_cnn_dft', 'all'])
    parser.add_argument('--data_path', type=str, default='data/raw/beta')
    parser.add_argument('--n_subjects', type=int, default=70)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--t_start', type=float, default=0.14)
    parser.add_argument('--t_end', type=float, default=2.14,
                        help='2s window (fit trong 3s BETA trial)')
    parser.add_argument('--save_dir', type=str, default='results/baselines_beta')
    parser.add_argument('--n_folds', type=int, default=None,
                        help='None=full 70 folds. 1=test nhanh')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    ) if args.device == 'auto' else torch.device(args.device)
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': 1e-4,
    }

    # Load BETA
    logger.info(f"Loading BETA dataset từ {args.data_path} ...")
    loader = BETADataset(
        data_path=args.data_path,
        t_start=args.t_start,
        t_end=args.t_end,
        verbose=False
    )
    X, y, subject_ids = loader.load_all_subjects(n_subjects=args.n_subjects)
    logger.info(f"BETA: X={X.shape}, subjects={len(np.unique(subject_ids))}")

    ext_list = BETA_EXTRACTORS if args.extractor == 'all' else [args.extractor]
    all_results = {}

    for ext in ext_list:
        result = train_loso_beta(
            ext, X, y, subject_ids, config, device, save_dir, args.n_folds
        )
        all_results[ext] = result

    # Summary
    print("\n" + "="*60)
    print("BETA LOSO — SUMMARY")
    print("="*60)
    print(f"{'Extractor':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-"*60)
    for name, res in all_results.items():
        print(f"{name:<25} {res['mean_acc']:>8.4f} {res['std_acc']:>8.4f} "
              f"{res['min_acc']:>8.4f} {res['max_acc']:>8.4f}")
    print("="*60)

    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out = save_dir / f"all_extractors_beta_loso_{ts}.json"
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"All results: {out}")


if __name__ == "__main__":
    main()
