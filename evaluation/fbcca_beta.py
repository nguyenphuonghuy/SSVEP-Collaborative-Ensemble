"""
evaluation/fbcca_beta.py
========================
Same-pipeline FBCCA on BETA (reviewer request: controlled baseline on the
second dataset). Reuses the exact FBCCA implementation of
evaluation/fbcca_baseline.py (a=1.25, b=0.25, 5 sub-bands [8k-2, 90] Hz,
5 harmonics) and the BETA loader used by the neural models (same 9 channels,
onset-relative window, 6-90 Hz band-pass, z-normalization).

BETA's target order differs from Benchmark's, so the stimulation
frequencies are read from the dataset's own suppl_info.freqs.

Usage:
    python evaluation/fbcca_beta.py --t_start 0.14 --t_end 2.14 \
        --save_dir results/fbcca_beta_fixedwindow_round
"""
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.beta_loader import BETADataset
from evaluation.fbcca_baseline import (build_reference_signals, make_filter_bank,
                                       fbcca_classify_trial)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def beta_freqs(data_path: str) -> np.ndarray:
    d = sio.loadmat(str(Path(data_path) / 'S1.mat'), squeeze_me=True, struct_as_record=False)
    return np.asarray(d['data'].suppl_info.freqs, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', default='data/raw/beta')
    ap.add_argument('--n_subjects', type=int, default=70)
    ap.add_argument('--t_start', type=float, default=0.14)
    ap.add_argument('--t_end', type=float, default=2.14)
    ap.add_argument('--sfreq', type=float, default=250.0)
    ap.add_argument('--n_harmonics', type=int, default=5)
    ap.add_argument('--n_subbands', type=int, default=5)
    ap.add_argument('--save_dir', default='results/fbcca_beta_fixedwindow_round')
    args = ap.parse_args()

    freqs = beta_freqs(args.data_path)
    log.info(f"BETA target freqs (dataset order): {np.round(freqs, 1).tolist()}")
    loader = BETADataset(data_path=args.data_path, t_start=args.t_start, t_end=args.t_end, verbose=False)
    X, y, subj = loader.load_all_subjects(n_subjects=args.n_subjects)
    n_samples = X.shape[2]
    log.info(f"Data X={X.shape}")

    refs = build_reference_signals(freqs, args.n_harmonics, n_samples, args.sfreq)
    fb = make_filter_bank(args.n_subbands, args.sfreq)

    per_subject, preds_all = [], {}
    for s in np.unique(subj):
        m = subj == s
        pred = np.array([fbcca_classify_trial(x, refs, fb) for x in X[m]])
        acc = float((pred == y[m]).mean())
        per_subject.append(acc)
        preds_all[int(s)] = pred.tolist()
        log.info(f"  S{s:02d} acc={acc:.4f}")

    res = {'method': 'FBCCA', 'dataset': 'BETA',
           'mean_acc': float(np.mean(per_subject)), 'std_acc': float(np.std(per_subject)),
           'per_subject_acc': per_subject, 'n_subjects': len(per_subject),
           'freqs': freqs.tolist(), 'config': vars(args), 'per_trial_pred': preds_all}
    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"fbcca_beta_results_{datetime.now():%Y%m%d_%H%M}.json"
    json.dump(res, open(f, 'w'), indent=1)
    log.info(f"FBCCA BETA mean={res['mean_acc']:.4f} +/- {res['std_acc']:.4f} -> {f}")


if __name__ == '__main__':
    main()
