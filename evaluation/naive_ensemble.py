"""
evaluation/naive_ensemble.py
=============================
Task 3: Naive Ensemble — Average logits từ 4 pretrained extractors.

Không có training mới. Load checkpoints đã train từ Task 2,
average softmax probabilities → classify.

Công thức:
    P_ensemble(x) = (1/4) * Σ softmax(extractor_i(x))
    y_pred = argmax(P_ensemble(x))

Đây là baseline quan trọng nhất:
- Nếu naive ensemble > best single (72.5%) → Criterion 3 ✓
- Nếu không → có vấn đề về diversity, cần xem lại

Usage:
    python evaluation/naive_ensemble.py \
        --data_path data/raw/benchmark \
        --checkpoint_dir results/baselines
"""

import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
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

# 4 extractors active trong ensemble
ENSEMBLE_EXTRACTORS = [
    'temporal_cnn',
    'freq_domain_cnn',
    # 'shallow_conv_net',  # Ablation: removed
    'spatial_temporal_cnn',
]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD PRETRAINED EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def load_extractor(
    name: str,
    fold: int,
    checkpoint_dir: Path,
    n_channels: int,
    n_samples: int,
    n_classes: int,
    device: torch.device
) -> torch.nn.Module:
    """Load pretrained extractor từ checkpoint."""
    ckpt_path = checkpoint_dir / f"{name}_fold{fold}.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint không tồn tại: {ckpt_path}\n"
            f"Hãy chạy training/train_extractors.py trước."
        )

    model = build_extractor(name, n_channels, n_samples, n_classes)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# NAIVE ENSEMBLE INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def naive_ensemble_predict(
    models: List[torch.nn.Module],
    X: torch.Tensor,
    device: torch.device,
    batch_size: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Naive ensemble: average softmax probabilities từ tất cả extractors.

    Returns:
        y_pred: (n_samples,) predicted class indices
        probs:  (n_samples, n_classes) averaged probabilities
    """
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []

    for (batch_x,) in loader:
        batch_x = batch_x.to(device)

        # Collect softmax từ mỗi extractor
        batch_probs = []
        for model in models:
            logits = model(batch_x)                      # (batch, n_classes)
            prob = F.softmax(logits, dim=-1)             # (batch, n_classes)
            batch_probs.append(prob)

        # Average across extractors
        # Shape: (n_extractors, batch, n_classes) → mean over dim 0
        avg_prob = torch.stack(batch_probs, dim=0).mean(dim=0)  # (batch, n_classes)
        all_probs.append(avg_prob.cpu())

    probs = torch.cat(all_probs, dim=0).numpy()          # (n_samples, n_classes)
    y_pred = probs.argmax(axis=1)                        # (n_samples,)
    return y_pred, probs


# ─────────────────────────────────────────────────────────────────────────────
# LOSO EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_naive_ensemble_loso(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    checkpoint_dir: Path,
    device: torch.device,
    batch_size: int = 64
) -> Dict:
    """
    LOSO evaluation của naive ensemble.
    Mỗi fold: load 4 checkpoints đã train trên 34 subjects,
    evaluate trên subject còn lại.
    """
    n_channels = X.shape[1]
    n_samples  = X.shape[2]
    n_classes  = len(np.unique(y))
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)

    logger.info(f"\n{'='*60}")
    logger.info(f"NAIVE ENSEMBLE — LOSO Evaluation")
    logger.info(f"  Extractors: {ENSEMBLE_EXTRACTORS}")
    logger.info(f"  Subjects: {n_subjects}, Classes: {n_classes}")
    logger.info(f"  Input: ({n_channels}ch × {n_samples}samples)")
    logger.info(f"{'='*60}")

    fold_results = []

    for fold_idx, test_subj in enumerate(unique_subjects):
        fold_num = fold_idx + 1

        # Split data
        test_mask  = subject_ids == test_subj
        X_test = torch.FloatTensor(X[test_mask])
        y_test = y[test_mask]

        # Load 4 pretrained models cho fold này
        models = []
        missing = []
        for ext_name in ENSEMBLE_EXTRACTORS:
            try:
                m = load_extractor(
                    ext_name, fold_num, checkpoint_dir,
                    n_channels, n_samples, n_classes, device
                )
                models.append(m)
            except FileNotFoundError as e:
                missing.append(ext_name)
                logger.warning(str(e))

        if len(missing) > 0:
            logger.error(f"Fold {fold_num}: thiếu checkpoints {missing} — skip")
            continue

        # Naive ensemble predict
        y_pred, probs = naive_ensemble_predict(models, X_test, device, batch_size)

        # Accuracy
        acc = (y_pred == y_test).mean()

        # Per-extractor accuracy (để so sánh)
        per_ext_accs = {}
        with torch.no_grad():
            X_test_gpu = X_test.to(device)
            for ext_name, model in zip(ENSEMBLE_EXTRACTORS, models):
                logits = model(X_test_gpu)
                pred = logits.argmax(dim=1).cpu().numpy()
                per_ext_accs[ext_name] = float((pred == y_test).mean())

        fold_results.append({
            'subject': int(test_subj),
            'fold': fold_num,
            'ensemble_acc': float(acc),
            'per_extractor': per_ext_accs,
            'best_single_acc': max(per_ext_accs.values()),
            'ensemble_gain': float(acc) - max(per_ext_accs.values()),
        })

        logger.info(
            f"  Fold {fold_num:2d}/35 | S{test_subj:02d} | "
            f"Ensemble={acc:.4f} | "
            f"Best single={max(per_ext_accs.values()):.4f} | "
            f"Gain={float(acc)-max(per_ext_accs.values()):+.4f}"
        )

    # ── Summary ──
    ensemble_accs   = [r['ensemble_acc']    for r in fold_results]
    best_single_accs = [r['best_single_acc'] for r in fold_results]
    gains            = [r['ensemble_gain']   for r in fold_results]

    mean_ensemble = np.mean(ensemble_accs)
    mean_best_single = np.mean(best_single_accs)
    mean_gain = np.mean(gains)

    # Per-extractor mean across folds
    per_ext_means = {}
    for ext_name in ENSEMBLE_EXTRACTORS:
        accs = [r['per_extractor'][ext_name] for r in fold_results]
        per_ext_means[ext_name] = float(np.mean(accs))

    result = {
        'method': 'naive_ensemble',
        'ensemble_mean_acc': float(mean_ensemble),
        'ensemble_std_acc': float(np.std(ensemble_accs)),
        'ensemble_min_acc': float(np.min(ensemble_accs)),
        'ensemble_max_acc': float(np.max(ensemble_accs)),
        'best_single_mean_acc': float(mean_best_single),
        'mean_gain_over_best_single': float(mean_gain),
        'n_folds_positive_gain': int(sum(1 for g in gains if g > 0)),
        'n_subjects': len(fold_results),
        'per_extractor_means': per_ext_means,
        'folds': fold_results,
    }

    # ── Print summary ──
    logger.info(f"\n{'='*60}")
    logger.info(f"NAIVE ENSEMBLE — RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  Ensemble mean acc : {mean_ensemble:.4f} ± {np.std(ensemble_accs):.4f}")
    logger.info(f"  Best single mean  : {mean_best_single:.4f}")
    logger.info(f"  Mean gain         : {mean_gain:+.4f}")
    logger.info(f"  Folds with gain>0 : {result['n_folds_positive_gain']}/{len(fold_results)}")
    logger.info(f"\n  Per-extractor means (trong fold test):")
    for ext, acc in per_ext_means.items():
        logger.info(f"    {ext:<25}: {acc:.4f}")

    logger.info(f"\n{'='*60}")
    logger.info(f"  Criterion 3 Check:")
    criterion3 = mean_ensemble > mean_best_single
    logger.info(f"  Ensemble ({mean_ensemble:.4f}) > Best single ({mean_best_single:.4f}): "
                f"{'✓ PASS' if criterion3 else '✗ FAIL'}")

    # So sánh với LOSO individual
    ext4_loso = 0.7245  # từ kết quả đã train
    criterion3_vs_ext4 = mean_ensemble > ext4_loso
    logger.info(f"  Ensemble ({mean_ensemble:.4f}) > Ext4 LOSO ({ext4_loso:.4f}): "
                f"{'✓ PASS' if criterion3_vs_ext4 else '✗ FAIL'}")
    logger.info(f"{'='*60}\n")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Naive Ensemble Evaluation')
    parser.add_argument('--dataset', type=str, default='benchmark',
                        choices=['benchmark', 'beta'],
                        help='Dataset de danh gia: benchmark hoac beta')
    parser.add_argument('--data_path', type=str, default='data/raw/benchmark')
    parser.add_argument('--checkpoint_dir', type=str, default='results/baselines')
    parser.add_argument('--save_dir', type=str, default='results/ensemble')
    parser.add_argument('--n_subjects', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--t_start', type=float, default=0.14)
    parser.add_argument('--t_end', type=float, default=2.14)  # fixed window bug: onset-relative now, 2.14 -> 2.0s
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Load data
    logger.info(f"Loading {args.dataset} dataset tu {args.data_path} ...")
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
    logger.info(f"Data: X={X.shape}, subjects={len(np.unique(subject_ids))}")

    # Evaluate
    checkpoint_dir = Path(args.checkpoint_dir)
    result = evaluate_naive_ensemble_loso(
        X=X,
        y=y,
        subject_ids=subject_ids,
        checkpoint_dir=checkpoint_dir,
        device=device,
        batch_size=args.batch_size,
    )

    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    save_path = save_dir / f"naive_ensemble_loso_{timestamp}.json"
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved: {save_path}")

    # Final verdict
    print("\n" + "="*65)
    print("NAIVE ENSEMBLE — FINAL SUMMARY")
    print("="*65)
    print(f"  Ensemble mean acc  : {result['ensemble_mean_acc']:.4f} "
          f"± {result['ensemble_std_acc']:.4f}")
    print(f"  Best single (fold) : {result['best_single_mean_acc']:.4f}")
    print(f"  Mean gain          : {result['mean_gain_over_best_single']:+.4f}")
    print(f"  Folds gain > 0     : {result['n_folds_positive_gain']}/35")
    print()
    print(f"  Criterion 3 (Ensemble > Best Single):", end=" ")
    if result['ensemble_mean_acc'] > result['best_single_mean_acc']:
        print("✓ PASS → Tiến vào Task 4 (Collaborative)")
    else:
        print("✗ FAIL → Cần phân tích thêm")
    print("="*65)


if __name__ == "__main__":
    main()
