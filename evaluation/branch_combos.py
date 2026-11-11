"""
evaluation/branch_combos.py
===========================
Offline analysis from the already-trained LOSO checkpoints (no new training).
For each fold, loads the three independently trained branches, computes the
per-trial softmax of each branch on the held-out subject, and evaluates all
1-, 2- and 3-branch probability-averaging combinations.

Saves:
  <save_dir>/<tag>_probs.npz   per-trial probabilities, labels and subject ids
  <save_dir>/<tag>_combos.json per-subject accuracy of every combination

Usage:
    python evaluation/branch_combos.py --dataset benchmark \
        --checkpoint_dir results/baselines_full_fixedwindow_round \
        --t_start 0.14 --t_end 5.14 --tag bm_5s
"""
import sys
import json
import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.extractors.extractors import build_extractor
from data.benchmark_loader import BenchmarkDataset
from data.beta_loader import BETADataset

BRANCHES = ['temporal_cnn', 'freq_domain_cnn', 'spatial_temporal_cnn']


@torch.no_grad()
def branch_probs(name, fold, ckdir, X, n_classes, device):
    m = build_extractor(name, X.shape[1], X.shape[2], n_classes)
    m.load_state_dict(torch.load(ckdir / f"{name}_fold{fold}.pt", map_location=device))
    m.to(device).eval()
    out = []
    for i in range(0, len(X), 128):
        out.append(F.softmax(m(torch.from_numpy(X[i:i + 128]).float().to(device)), -1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['benchmark', 'beta'], default='benchmark')
    ap.add_argument('--data_path', default=None)
    ap.add_argument('--checkpoint_dir', required=True)
    ap.add_argument('--t_start', type=float, default=0.14)
    ap.add_argument('--t_end', type=float, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--save_dir', default='results/branch_combos')
    a = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if a.dataset == 'beta':
        ld = BETADataset(data_path=a.data_path or 'data/raw/beta', t_start=a.t_start, t_end=a.t_end, verbose=False)
        X, y, subj = ld.load_all_subjects(n_subjects=70)
    else:
        ld = BenchmarkDataset(data_path=a.data_path or 'data/raw/benchmark', t_start=a.t_start, t_end=a.t_end, verbose=False)
        X, y, subj = ld.load_all_subjects(n_subjects=35)
    n_classes = len(np.unique(y))
    ck = Path(a.checkpoint_dir)

    P = np.zeros((len(BRANCHES), len(y), n_classes), dtype=np.float32)
    for fold, s in enumerate(np.unique(subj), start=1):
        m = subj == s
        for b, name in enumerate(BRANCHES):
            P[b, m] = branch_probs(name, fold, ck, X[m], n_classes, dev)
        print(f"fold {fold} S{s}", flush=True)

    combos = {}
    for r in (1, 2, 3):
        for c in itertools.combinations(range(3), r):
            pred = P[list(c)].mean(0).argmax(-1)
            key = '+'.join(BRANCHES[i] for i in c)
            combos[key] = [float((pred[subj == s] == y[subj == s]).mean()) for s in np.unique(subj)]
    out = Path(a.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{a.tag}_probs.npz", P=P, y=y, subj=subj, branches=np.array(BRANCHES))
    json.dump({'tag': a.tag, 'config': vars(a), 'subjects': np.unique(subj).tolist(),
               'per_subject_acc': combos,
               'mean_acc': {k: float(np.mean(v)) for k, v in combos.items()}},
              open(out / f"{a.tag}_combos.json", 'w'), indent=1)
    for k, v in combos.items():
        print(f"{k:60s} {np.mean(v)*100:6.2f}")


if __name__ == '__main__':
    main()
