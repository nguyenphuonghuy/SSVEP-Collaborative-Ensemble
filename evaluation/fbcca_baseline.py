"""
evaluation/fbcca_baseline.py
==============================
Baseline FBCCA (filter-bank canonical correlation analysis) cho Muc 5.3.2
(ablation nguon goc dong gop cua "harmonic stacking").

Phuong phap training-free kinh dien, dung de doi chieu voi cac bien the
hoc duoc (harmonic_amp_linear, harmonic_stack_linear, freq_domain_cnn).

Cong thuc dung dung nhu \\eqref{eq:fbcca} trong bai bao:
    rho' = sum_n w(n) * (rho^(n))^2,    w(n) = n^{-a} + b
voi rho^(n) la canonical correlation cua sub-band thu n giua tin hieu EEG
da loc va tin hieu tham chieu sin-cos tai tan so kich thich + cac hai.

Hang so a=1.25, b=0.25 lay tu Chen et al. 2015 "Filter bank canonical
correlation analysis for implementing a high-speed SSVEP-based
brain-computer interface", J. Neural Eng. - cach chon pho bien nhat trong
tai lieu FBCCA, khong tu dieu chinh rieng cho nghien cuu nay.

Usage:
    python evaluation/fbcca_baseline.py \\
        --data_path data/raw/benchmark --t_start 0.14 --t_end 4.14
"""

import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import signal, linalg

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.benchmark_loader import BenchmarkDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

TARGET_FREQS = np.array([
    8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2,
    8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4,
    8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6,
    8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8,
])


def canonical_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """
    He so tuong quan chinh tac lon nhat giua X (n_samples, p) va
    Y (n_samples, q), dung QR + SVD (Hotelling CCA co dien).
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    Qx, Rx = linalg.qr(X, mode='economic')
    Qy, Ry = linalg.qr(Y, mode='economic')

    # Rank deficiency guard (tin hieu hang hoac gan hang)
    if np.linalg.matrix_rank(Rx) < Rx.shape[1] or np.linalg.matrix_rank(Ry) < Ry.shape[1]:
        Rx_pinv_rank = min(Qx.shape[1], Qy.shape[1])
    try:
        M = Qx.T @ Qy
        s = linalg.svd(M, compute_uv=False)
        return float(s[0]) if len(s) > 0 else 0.0
    except np.linalg.LinAlgError:
        return 0.0


def build_reference_signals(
    freqs: np.ndarray, n_harmonics: int, n_samples: int, sfreq: float
) -> np.ndarray:
    """Returns: (n_targets, n_samples, 2*n_harmonics) - cap sin/cos moi hai."""
    t = np.arange(n_samples) / sfreq
    n_targets = len(freqs)
    refs = np.zeros((n_targets, n_samples, 2 * n_harmonics))
    for i, f in enumerate(freqs):
        for h in range(1, n_harmonics + 1):
            refs[i, :, 2 * (h - 1)]     = np.sin(2 * np.pi * f * h * t)
            refs[i, :, 2 * (h - 1) + 1] = np.cos(2 * np.pi * f * h * t)
    return refs


def make_filter_bank(n_subbands: int, sfreq: float, low_base: float = 8.0,
                      high: float = 90.0):
    """
    Sub-band thu k (k=1..n_subbands) co dai thong [k*low_base - 2, high] Hz,
    theo thiet ke chuan cua Chen et al. 2015 (moi sub-band bat dau tu hai
    thu k cua tan so kich thich thap nhat).
    """
    filters = []
    nyq = sfreq / 2
    for k in range(1, n_subbands + 1):
        low = max(k * low_base - 2, 1.0)
        high_k = min(high, nyq - 1)
        if low >= high_k:
            low = high_k - 5
        sos = signal.butter(4, [low / nyq, high_k / nyq], btype='band', output='sos')
        filters.append(sos)
    return filters


def fbcca_classify_trial(
    trial: np.ndarray,           # (n_channels, n_samples)
    ref_signals: np.ndarray,     # (n_targets, n_samples, 2*n_harmonics)
    filter_bank: list,
    a: float = 1.25, b: float = 0.25
) -> int:
    n_subbands = len(filter_bank)
    n_targets = ref_signals.shape[0]
    weights = np.array([n ** (-a) + b for n in range(1, n_subbands + 1)])

    scores = np.zeros(n_targets)
    for sb_idx, sos in enumerate(filter_bank):
        filtered = signal.sosfiltfilt(sos, trial, axis=-1)   # (n_ch, n_samples)
        X = filtered.T  # (n_samples, n_ch)
        for t_idx in range(n_targets):
            rho = canonical_correlation(X, ref_signals[t_idx])
            scores[t_idx] += weights[sb_idx] * (rho ** 2)

    return int(np.argmax(scores))


def main():
    parser = argparse.ArgumentParser(description='FBCCA baseline (training-free)')
    parser.add_argument('--data_path', type=str, default='data/raw/benchmark')
    parser.add_argument('--n_subjects', type=int, default=35)
    parser.add_argument('--t_start', type=float, default=0.14)
    parser.add_argument('--t_end', type=float, default=4.14)
    parser.add_argument('--sfreq', type=float, default=250.0)
    parser.add_argument('--n_harmonics', type=int, default=5)
    parser.add_argument('--n_subbands', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='results/fbcca')
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Benchmark dataset tu {args.data_path} ...")
    loader = BenchmarkDataset(
        data_path=args.data_path, t_start=args.t_start, t_end=args.t_end, verbose=False
    )
    X, y, subject_ids = loader.load_all_subjects(n_subjects=args.n_subjects)
    n_samples = X.shape[2]
    logger.info(f"Data: X={X.shape}, subjects={len(np.unique(subject_ids))}")

    ref_signals = build_reference_signals(
        TARGET_FREQS, args.n_harmonics, n_samples, args.sfreq
    )
    filter_bank = make_filter_bank(args.n_subbands, args.sfreq)
    logger.info(f"Filter bank: {args.n_subbands} sub-bands, "
                f"{args.n_harmonics} harmonics/target")

    unique_subjects = np.unique(subject_ids)
    per_subject_acc = []

    for s_idx, subj in enumerate(unique_subjects):
        mask = subject_ids == subj
        Xs, ys = X[mask], y[mask]
        correct = 0
        for i in range(len(ys)):
            pred = fbcca_classify_trial(Xs[i], ref_signals, filter_bank)
            correct += int(pred == ys[i])
        acc = correct / len(ys)
        per_subject_acc.append(acc)
        logger.info(f"  Subject {subj:2d} ({s_idx+1}/{len(unique_subjects)}) | "
                    f"n_trials={len(ys)} | Acc={acc:.4f}")

    mean_acc = float(np.mean(per_subject_acc))
    std_acc = float(np.std(per_subject_acc))

    result = {
        'method': 'FBCCA',
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'min_acc': float(np.min(per_subject_acc)),
        'max_acc': float(np.max(per_subject_acc)),
        'per_subject_acc': [float(a) for a in per_subject_acc],
        'n_subjects': len(unique_subjects),
        'config': vars(args),
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"FBCCA — KET QUA")
    logger.info(f"  Mean acc: {mean_acc:.4f} ± {std_acc:.4f}")
    logger.info(f"{'='*60}")

    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out = save_dir / f"fbcca_results_{ts}.json"
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved: {out}")


if __name__ == "__main__":
    main()
