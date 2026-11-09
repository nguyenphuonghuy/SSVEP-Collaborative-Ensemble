"""
training/train_extractors.py
=============================
Training pipeline cho individual extractors với LOSO cross-validation.
Phase 1: Validate từng extractor độc lập trước khi build ensemble.

Usage:
    python training/train_extractors.py --extractor temporal_cnn --use_dummy
    python training/train_extractors.py --extractor all --data_path data/raw/benchmark
"""

import os
import sys
import json
import time
import random
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.extractors.extractors import build_extractor, build_all_extractors, EXTRACTOR_REGISTRY
from data.benchmark_loader import BenchmarkDataset, create_dummy_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CORE
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
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
    
    return {
        'loss': total_loss / total,
        'acc': correct / total
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
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
    
    return {
        'loss': total_loss / total,
        'acc': correct / total
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOSO TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_loso(
    extractor_name: str,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    config: dict,
    device: torch.device,
    save_dir: Path
) -> Dict:
    """
    Leave-One-Subject-Out training cho 1 extractor.
    
    Returns dict với per-fold và mean accuracy.
    """
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)
    
    n_channels = X.shape[1]
    n_samples = X.shape[2]
    n_classes = len(np.unique(y))
    
    fold_results = []
    
    logger.info(f"\n{'='*60}")
    logger.info(f"LOSO Training: {extractor_name}")
    logger.info(f"  Subjects: {n_subjects}, Classes: {n_classes}")
    logger.info(f"  Input: ({n_channels}ch × {n_samples}samples)")
    logger.info(f"{'='*60}")
    
    for fold_idx, test_subj in enumerate(unique_subjects):
        fold_start = time.time()

        # Resume: checkpoint da co san (vi du chay lai sau khi bi kill giua chung
        # do het walltime) -> danh gia lai thay vi train lai tu dau.
        ckpt_path = save_dir / f"{extractor_name}_fold{fold_idx+1}.pt"
        done_path = save_dir / f"{extractor_name}_fold{fold_idx+1}.done.json"
        if done_path.exists() and ckpt_path.exists():
            test_mask = subject_ids == test_subj
            X_test, y_test = X[test_mask], y[test_mask]
            test_ds = TensorDataset(
                torch.FloatTensor(X_test), torch.LongTensor(y_test)
            )
            test_loader = DataLoader(test_ds, batch_size=config['batch_size'],
                                      shuffle=False, num_workers=2, pin_memory=True)
            model = build_extractor(extractor_name, n_channels, n_samples, n_classes).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            resumed_stats = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
            fold_results.append({
                'subject': int(test_subj),
                'fold': fold_idx + 1,
                'best_acc': float(resumed_stats['acc']),
                'best_val_acc': None,
                'best_epoch': None,
                'time_sec': 0.0,
            })
            logger.info(
                f"  [RESUME] Fold {fold_idx+1} | Subject S{test_subj:02d} | "
                f"Test={resumed_stats['acc']:.4f} (bỏ qua, checkpoint đã có từ trước)"
            )
            continue

        # Split: test subject hoan toan tach biet. Trong so 34 subject con lai,
        # tach them mot tap validation theo SUBJECT (khong dung trial cua test
        # subject) de chon checkpoint - test subject KHONG bao gio duoc dung
        # de ra bat ky quyet dinh nao (epoch/checkpoint), chi danh gia 1 lan
        # duy nhat sau khi huan luyen xong. Xem Muc 3.4 (validation split).
        train_pool = unique_subjects[unique_subjects != test_subj]
        rng = np.random.RandomState(1000 + fold_idx)
        shuffled = rng.permutation(train_pool)
        n_val = min(max(3, round(0.15 * len(train_pool))), max(1, len(train_pool) - 1))
        val_subjects = set(shuffled[:n_val].tolist())
        train_subjects = set(shuffled[n_val:].tolist())

        test_mask = subject_ids == test_subj
        val_mask = np.isin(subject_ids, list(val_subjects))
        train_mask = np.isin(subject_ids, list(train_subjects))

        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        # DataLoaders
        train_ds = TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val), torch.LongTensor(y_val)
        )
        test_ds = TensorDataset(
            torch.FloatTensor(X_test), torch.LongTensor(y_test)
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        # Model
        model = build_extractor(extractor_name, n_channels, n_samples, n_classes)
        model = model.to(device)

        # Optimizer & scheduler
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs']
        )
        criterion = nn.CrossEntropyLoss()

        # Mixed precision
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

        # Training loop - checkpoint duoc chon theo VALIDATION acc, khong phai
        # test acc (test subject khong tham gia bat ky quyet dinh nao o day).
        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(config['epochs']):
            train_stats = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler
            )
            val_stats = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if val_stats['acc'] > best_val_acc:
                best_val_acc = val_stats['acc']
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)

            # Log mỗi 20 epochs
            if (epoch + 1) % 20 == 0 or epoch == 0:
                logger.info(
                    f"  Fold {fold_idx+1}/{n_subjects} | "
                    f"Epoch {epoch+1:3d}/{config['epochs']} | "
                    f"Train acc={train_stats['acc']:.3f} | "
                    f"Val acc={val_stats['acc']:.3f} | "
                    f"Best(val)={best_val_acc:.3f}"
                )

        # Danh gia TREN TEST DUNG MOT LAN DUY NHAT, dung checkpoint da chon
        # boi validation - day la con so duoc bao cao.
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        final_test_stats = evaluate(model, test_loader, criterion, device)
        best_test_acc = final_test_stats['acc']

        fold_time = time.time() - fold_start
        fold_results.append({
            'subject': int(test_subj),
            'fold': fold_idx + 1,
            'best_acc': best_test_acc,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'time_sec': fold_time
        })
        with open(done_path, 'w') as f:
            json.dump(fold_results[-1], f)

        logger.info(
            f"  ✓ Fold {fold_idx+1} done | "
            f"Subject S{test_subj:02d} | "
            f"Val={best_val_acc:.4f} → Test={best_test_acc:.4f} ({fold_time:.1f}s)"
        )
    
    # Summary
    accs = [r['best_acc'] for r in fold_results]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    
    result = {
        'extractor': extractor_name,
        'mean_acc': float(mean_acc),
        'std_acc': float(std_acc),
        'min_acc': float(np.min(accs)),
        'max_acc': float(np.max(accs)),
        'n_subjects': n_subjects,
        'folds': fold_results,
        'config': config
    }
    
    logger.info(f"\n{'='*60}")
    logger.info(f"LOSO Result: {extractor_name}")
    logger.info(f"  Mean acc: {mean_acc:.4f} ± {std_acc:.4f}")
    logger.info(f"  Range: [{np.min(accs):.4f}, {np.max(accs):.4f}]")
    logger.info(f"{'='*60}\n")
    
    # Save results
    result_path = save_dir / f"{extractor_name}_loso_results.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"  Results saved: {result_path}")
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train SSVEP feature extractors')
    parser.add_argument('--extractor', type=str, default='all',
                        choices=list(EXTRACTOR_REGISTRY.keys()) + ['all'],
                        help='Extractor to train (default: all)')
    parser.add_argument('--data_path', type=str, default='data/raw/benchmark',
                        help='Path to Benchmark dataset')
    parser.add_argument('--use_dummy', action='store_true',
                        help='Dùng dummy data để test pipeline')
    parser.add_argument('--n_subjects', type=int, default=35,
                        help='Số subjects (35 cho full dataset)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs per fold')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--t_start', type=float, default=0.14,
                        help='Trial start time (giây)')
    parser.add_argument('--t_end', type=float, default=2.14,
                        help='Trial end time (giây, tính từ onset kích thích thực; 2.14 -> 2.0s)')
    parser.add_argument('--save_dir', type=str, default='results/baselines')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=None,
                        help='multi-seed control: seed khoi tao trong so + thu '
                             'tu shuffle batch cho giai doan baseline (mac '
                             'dinh None = khong seed, hanh vi cu giu nguyen). '
                             'Dat gia tri cu the de lap lai giai doan baseline '
                             'voi nhieu seed doc lap, tach bach nhieu khoi tao '
                             'ngau nhien khoi bien thien giua cac doi tuong.')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {args.seed if args.seed is not None else 'unseeded (hanh vi mac dinh)'}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Save dir
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Config
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': 1e-4,
    }
    
    # Load data
    if args.use_dummy:
        logger.info("⚠ Dùng DUMMY DATA (để test pipeline)")
        n_dummy = min(args.n_subjects, 3)  # dummy chỉ cần 3 subjects
        X, y, subject_ids = create_dummy_data(n_subjects=n_dummy)
        config['epochs'] = min(args.epochs, 20)  # giảm epochs cho dummy
    else:
        logger.info(f"Loading Benchmark dataset từ {args.data_path}")
        loader = BenchmarkDataset(
            data_path=args.data_path,
            t_start=args.t_start,
            t_end=args.t_end,
        )
        X, y, subject_ids = loader.load_all_subjects(n_subjects=args.n_subjects)
    
    logger.info(f"Data loaded: X={X.shape}, y={y.shape}, "
                f"subjects={len(np.unique(subject_ids))}")
    
    # Xác định extractors cần train
    if args.extractor == 'all':
        extractor_names = list(EXTRACTOR_REGISTRY.keys())
    else:
        extractor_names = [args.extractor]
    
    # Train
    all_results = {}
    
    for ext_name in extractor_names:
        logger.info(f"\n>>> Training: {ext_name}")
        result = train_loso(
            extractor_name=ext_name,
            X=X,
            y=y,
            subject_ids=subject_ids,
            config=config,
            device=device,
            save_dir=save_dir
        )
        all_results[ext_name] = result
    
    # Final summary
    print("\n" + "="*65)
    print("SUMMARY - Individual Extractor LOSO Results")
    print("="*65)
    print(f"{'Extractor':<25} {'Mean Acc':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-"*65)
    for name, res in all_results.items():
        print(f"{name:<25} {res['mean_acc']:>10.4f} "
              f"{res['std_acc']:>8.4f} "
              f"{res['min_acc']:>8.4f} "
              f"{res['max_acc']:>8.4f}")
    print("="*65)
    
    # Check criterion 2
    print("\n📊 Check Success Criterion 2 (vs EEGNet ~65%):")
    eegnet_baseline = 0.65  # từ handoff doc, thay bằng kết quả thực sau baseline
    for name, res in all_results.items():
        diff = res['mean_acc'] - eegnet_baseline
        status = "✓" if diff >= -0.02 else "✗"
        print(f"  {status} {name}: {res['mean_acc']:.4f} "
              f"(EEGNet {'+' if diff >= 0 else ''}{diff:.4f})")
    
    # Save combined results
    combined_path = save_dir / f"all_extractors_loso_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nAll results saved: {combined_path}")


if __name__ == "__main__":
    main()
