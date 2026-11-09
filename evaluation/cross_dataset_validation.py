"""
evaluation/cross_dataset_validation.py
========================================
Task 6: Cross-dataset validation.

Protocol:
  Train: Benchmark (35 subjects) — dùng checkpoints đã train
  Test : BETA (70 subjects) — zero-shot inference

2 modes:
  A) Zero-shot: dùng thẳng Benchmark checkpoints → test trên BETA
     Đo generalization mà không cần bất kỳ BETA data nào
  B) Few-shot: fine-tune nhẹ trên 10% BETA (7 subjects) → test 63 subjects
     Đo khả năng adapt nhanh

Usage:
    python evaluation/cross_dataset_validation.py \
        --benchmark_collab_dir results/ensemble_4s_3ext \
        --beta_data_path data/raw/beta \
        --mode zero_shot
"""

import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.extractors.extractors import build_extractor
from data.beta_loader import BETADataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 3-extractor ensemble (kết quả tốt nhất từ ablation)
ENSEMBLE_EXTRACTORS = [
    'temporal_cnn',
    'freq_domain_cnn',
    'spatial_temporal_cnn',
]


def load_benchmark_models(
    collab_dir: Path,
    fold: int,
    n_ch: int,
    n_samp: int,
    n_cls: int,
    device: torch.device,
    benchmark_n_samp: int = 1000   # Benchmark 4s = 1000 samples
) -> List[torch.nn.Module]:
    """
    Load collaborative models từ 1 fold Benchmark.
    Quan trọng: model phải được khởi tạo với benchmark_n_samp (1000),
    không phải BETA n_samp (715) — vì checkpoint được train với Benchmark shape.
    """
    models = []
    for ext in ENSEMBLE_EXTRACTORS:
        # Thử collab checkpoint trước, fallback sang baseline
        ckpt = collab_dir / f"{ext}_collab_fold{fold}.pt"
        if not ckpt.exists():
            ckpt = collab_dir / f"{ext}_fold{fold}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint không tìm thấy: {ckpt}")

        # Khởi tạo model với Benchmark n_samples (1000), không phải BETA (715)
        m = build_extractor(ext, n_ch, benchmark_n_samp, n_cls)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.to(device).eval()
        models.append(m)
    return models


BENCHMARK_N_SAMP = 1360   # Benchmark 4s @ 250Hz


def pad_to_length(X: np.ndarray, target_len: int) -> np.ndarray:
    """
    Pad hoặc truncate time dimension của X về target_len.
    X: (n_trials, n_channels, n_samples)
    """
    current_len = X.shape[2]
    if current_len == target_len:
        return X
    elif current_len < target_len:
        # Zero-pad bên phải
        pad = np.zeros((X.shape[0], X.shape[1], target_len - current_len),
                       dtype=X.dtype)
        return np.concatenate([X, pad], axis=2)
    else:
        # Truncate
        return X[:, :, :target_len]


@torch.no_grad()
def ensemble_predict(models, X, device, batch_size=64):
    """Ensemble predict → probability array."""
    ds = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_probs = []
    for (xb,) in loader:
        xb = xb.to(device)
        probs = torch.stack([F.softmax(m(xb), dim=-1)
                             for m in models]).mean(0)
        all_probs.append(probs.cpu())
    return torch.cat(all_probs).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# MODE A: Zero-shot
# ─────────────────────────────────────────────────────────────────────────────

def zero_shot_validation(
    X_beta: np.ndarray,
    y_beta: np.ndarray,
    subject_ids_beta: np.ndarray,
    collab_dir: Path,
    device: torch.device,
    n_ch: int, n_samp: int, n_cls: int,
    n_benchmark_folds: int = 35,
    batch_size: int = 64
) -> Dict:
    """
    Zero-shot: dùng checkpoints Benchmark, test trực tiếp trên BETA.
    BETA data (715 samples) được pad lên 1000 samples để match Benchmark model.
    """
    logger.info("\n=== Zero-shot Cross-dataset Validation ===")
    logger.info(f"  BETA subjects: {len(np.unique(subject_ids_beta))}")
    logger.info(f"  BETA n_samples={n_samp} → pad lên {BENCHMARK_N_SAMP}")
    logger.info(f"  Dùng Benchmark fold 1 checkpoints")

    # Pad BETA lên Benchmark n_samples
    X_beta_padded = pad_to_length(X_beta, BENCHMARK_N_SAMP)
    logger.info(f"  Padded BETA: {X_beta_padded.shape}")

    # Load models với Benchmark n_samples
    try:
        models = load_benchmark_models(
            collab_dir, fold=1,
            n_ch=n_ch, n_samp=BENCHMARK_N_SAMP, n_cls=n_cls,
            device=device, benchmark_n_samp=BENCHMARK_N_SAMP
        )
        logger.info(f"  Loaded {len(models)} models từ fold 1")
    except FileNotFoundError as e:
        logger.error(str(e))
        return {}

    unique_subjects = np.unique(subject_ids_beta)
    fold_results = []

    for subj in unique_subjects:
        mask = subject_ids_beta == subj
        X_test = X_beta_padded[mask]
        y_test = y_beta[mask]

        probs = ensemble_predict(models, X_test, device, batch_size)
        acc = (probs.argmax(1) == y_test).mean()

        fold_results.append({
            'subject': int(subj),
            'n_trials': int(len(y_test)),
            'accuracy': float(acc),
        })
        logger.info(f"  BETA S{subj:02d}: acc={acc:.4f} ({len(y_test)} trials)")

    accs = [r['accuracy'] for r in fold_results]
    result = {
        'mode': 'zero_shot',
        'mean_acc': float(np.mean(accs)),
        'std_acc': float(np.std(accs)),
        'min_acc': float(np.min(accs)),
        'max_acc': float(np.max(accs)),
        'n_subjects': len(fold_results),
        'subjects': fold_results,
    }

    logger.info(f"\n  Zero-shot BETA: {result['mean_acc']:.4f} ± {result['std_acc']:.4f}")
    logger.info(f"  Range: [{result['min_acc']:.4f}, {result['max_acc']:.4f}]")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MODE B: Few-shot fine-tune
# ─────────────────────────────────────────────────────────────────────────────

def few_shot_validation(
    X_beta: np.ndarray,
    y_beta: np.ndarray,
    subject_ids_beta: np.ndarray,
    collab_dir: Path,
    device: torch.device,
    n_ch: int, n_samp: int, n_cls: int,
    n_finetune_subjects: int = 7,   # 10% của 70
    finetune_epochs: int = 20,
    lr: float = 0.0001,
    batch_size: int = 16
) -> Dict:
    """
    Few-shot: fine-tune trên n_finetune_subjects BETA subjects,
    test trên phần còn lại.
    """
    import torch.optim as optim

    logger.info(f"\n=== Few-shot Cross-dataset Validation ===")
    logger.info(f"  Fine-tune: {n_finetune_subjects} subjects")
    logger.info(f"  Test: {70 - n_finetune_subjects} subjects")

    unique_subjects = np.unique(subject_ids_beta)
    finetune_subjs = unique_subjects[:n_finetune_subjects]
    test_subjs     = unique_subjects[n_finetune_subjects:]

    # Load models từ fold 1
    models = load_benchmark_models(
        collab_dir, fold=1,
        n_ch=n_ch, n_samp=n_samp, n_cls=n_cls, device=device
    )

    # Fine-tune trên BETA subjects đầu tiên
    finetune_mask = np.isin(subject_ids_beta, finetune_subjs)
    X_ft = X_beta[finetune_mask]
    y_ft = y_beta[finetune_mask]

    logger.info(f"  Fine-tune data: X={X_ft.shape}")

    ds_ft = TensorDataset(
        torch.FloatTensor(X_ft), torch.LongTensor(y_ft)
    )
    loader_ft = DataLoader(ds_ft, batch_size=batch_size,
                           shuffle=True, num_workers=0)

    optimizers = [
        optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
        for m in models
    ]
    ce = torch.nn.CrossEntropyLoss()

    for epoch in range(finetune_epochs):
        for m in models:
            m.train()
        for xb, yb in loader_ft:
            xb, yb = xb.to(device), yb.to(device)
            for m, opt in zip(models, optimizers):
                opt.zero_grad(set_to_none=True)
                loss = ce(m(xb), yb)
                loss.backward()
                opt.step()

        if (epoch + 1) % 5 == 0:
            # Quick eval trên fine-tune set
            for m in models:
                m.eval()
            with torch.no_grad():
                probs = ensemble_predict(models, X_ft, device, batch_size)
                ft_acc = (probs.argmax(1) == y_ft).mean()
            logger.info(f"  Epoch {epoch+1}/{finetune_epochs} | FT acc={ft_acc:.4f}")

    # Evaluate trên test subjects
    for m in models:
        m.eval()

    fold_results = []
    for subj in test_subjs:
        mask = subject_ids_beta == subj
        X_test = X_beta[mask]
        y_test = y_beta[mask]
        probs = ensemble_predict(models, X_test, device, batch_size)
        acc = (probs.argmax(1) == y_test).mean()
        fold_results.append({
            'subject': int(subj),
            'accuracy': float(acc),
        })
        logger.info(f"  BETA S{subj:02d}: acc={acc:.4f}")

    accs = [r['accuracy'] for r in fold_results]
    result = {
        'mode': 'few_shot',
        'n_finetune_subjects': n_finetune_subjects,
        'n_test_subjects': len(test_subjs),
        'mean_acc': float(np.mean(accs)),
        'std_acc': float(np.std(accs)),
        'min_acc': float(np.min(accs)),
        'max_acc': float(np.max(accs)),
        'subjects': fold_results,
    }

    logger.info(f"\n  Few-shot BETA: {result['mean_acc']:.4f} ± {result['std_acc']:.4f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Task 6: Cross-dataset Validation')
    parser.add_argument('--beta_data_path', type=str, default='data/raw/beta')
    parser.add_argument('--benchmark_collab_dir', type=str,
                        default='results/ensemble_4s_3ext')
    parser.add_argument('--save_dir', type=str, default='results/ablation')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['zero_shot', 'few_shot', 'both'])
    parser.add_argument('--n_subjects', type=int, default=70)
    parser.add_argument('--n_finetune', type=int, default=7,
                        help='Số subjects BETA dùng để fine-tune (few-shot mode)')
    parser.add_argument('--finetune_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    # BETA trial = 3s → t_end tối đa ~2.9s để lấy ~700 samples sau latency
    parser.add_argument('--t_start', type=float, default=0.14)
    parser.add_argument('--t_end', type=float, default=None,
                        help='None = dùng hết 3s trial. Hoặc set 2.14 để match 2s window')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    # Device
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    ) if args.device == 'auto' else torch.device(args.device)
    logger.info(f"Device: {device}")

    # Load BETA
    logger.info(f"Loading BETA dataset từ {args.beta_data_path} ...")
    loader = BETADataset(
        data_path=args.beta_data_path,
        t_start=args.t_start,
        t_end=args.t_end,
        verbose=False
    )
    X, y, subject_ids = loader.load_all_subjects(n_subjects=args.n_subjects)
    n_ch, n_samp, n_cls = X.shape[1], X.shape[2], len(np.unique(y))
    logger.info(f"BETA data: X={X.shape}, classes={n_cls}, "
                f"subjects={len(np.unique(subject_ids))}")

    collab_dir = Path(args.benchmark_collab_dir)
    save_dir   = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.mode in ('zero_shot', 'both'):
        res = zero_shot_validation(
            X, y, subject_ids, collab_dir, device,
            n_ch, n_samp, n_cls, batch_size=args.batch_size
        )
        all_results['zero_shot'] = res

    if args.mode in ('few_shot', 'both'):
        res = few_shot_validation(
            X, y, subject_ids, collab_dir, device,
            n_ch, n_samp, n_cls,
            n_finetune_subjects=args.n_finetune,
            finetune_epochs=args.finetune_epochs,
            batch_size=args.batch_size
        )
        all_results['few_shot'] = res

    # Save
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out = save_dir / f"cross_dataset_{ts}.json"
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved: {out}")

    # Summary
    print("\n" + "="*60)
    print("CROSS-DATASET VALIDATION — SUMMARY")
    print("="*60)
    if 'zero_shot' in all_results:
        r = all_results['zero_shot']
        print(f"  Zero-shot  BETA: {r['mean_acc']:.4f} ± {r['std_acc']:.4f}")
    if 'few_shot' in all_results:
        r = all_results['few_shot']
        print(f"  Few-shot   BETA: {r['mean_acc']:.4f} ± {r['std_acc']:.4f}")
    print(f"  Benchmark (ref): 0.9327")
    print("="*60)


if __name__ == "__main__":
    main()
