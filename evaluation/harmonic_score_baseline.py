"""
evaluation/harmonic_score_baseline.py
=====================================
Training-free harmonic scoring on the SAME stacked tensor that SSVEPFreqNet
receives (reviewer request: does the learned head add anything beyond a
fixed rule?). For each target n:

    score_n = sum_h w_h * sum_c A_c(h * f_n)

with A the per-channel amplitude spectrum, H = 5, and two weightings:
uniform (w_h = 1) and FBCCA-style (w_h = h^-1.25 + 0.25). Amplitudes are read
either at the nearest FFT bin (as SSVEPFreqNet) or exactly at h*f_n by a direct
DFT. Same loaders, channels and windows as the neural models.

Usage:
    python evaluation/harmonic_score_baseline.py --out results/phase2/harmonic_score.json
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.benchmark_loader import BenchmarkDataset
from data.beta_loader import BETADataset

BM_FREQS = np.array([
    8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2,
    8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4,
    8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6,
    8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8])
FS, H = 250.0, 5
OFFSET = 0.5  # nearest-bin rounding, as _FREQ_BIN_OFFSET in the extractors


def amplitudes(X, freqs, mode):
    """X: (n, C, T) -> (n, C, 40, H)"""
    T = X.shape[-1]
    if mode == 'fft':
        A = np.abs(np.fft.rfft(X, axis=-1))
        res = FS / T
        idx = np.floor(np.outer(freqs, np.arange(1, H + 1)) / res + OFFSET).astype(int)
        return A[:, :, idx]
    t = np.arange(T) / FS
    f = np.outer(freqs, np.arange(1, H + 1)).reshape(-1)
    E = np.exp(-2j * np.pi * f[:, None] * t[None, :])         # (40H, T)
    return np.abs(X @ E.T).reshape(X.shape[0], X.shape[1], len(freqs), H)


def evaluate(X, y, subj, freqs):
    out = {}
    w_fb = np.arange(1, H + 1) ** -1.25 + 0.25
    for mode in ('fft', 'dft'):
        S = amplitudes(X, freqs, mode).sum(1)                   # (n, 40, H)
        for wname, w in (('uniform', np.ones(H)), ('fbcca_w', w_fb)):
            pred = (S * w).sum(-1).argmax(-1)
            out[f'{mode}_{wname}'] = [float((pred[subj == s] == y[subj == s]).mean()) for s in np.unique(subj)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/phase2/harmonic_score.json')
    a = ap.parse_args()
    res = {}
    for L in (1, 2, 4, 5):
        ld = BenchmarkDataset(data_path='data/raw/benchmark', t_start=0.14, t_end=0.14 + L, verbose=False)
        X, y, s = ld.load_all_subjects(n_subjects=35)
        res[f'bm_{L}s'] = evaluate(X.astype(np.float64), y, s, BM_FREQS)
        print(L, {k: round(np.mean(v) * 100, 2) for k, v in res[f'bm_{L}s'].items()}, flush=True)
    d = sio.loadmat('data/raw/beta/S1.mat', squeeze_me=True, struct_as_record=False)
    bf = np.asarray(d['data'].suppl_info.freqs, dtype=float)
    ld = BETADataset(data_path='data/raw/beta', t_start=0.14, t_end=2.14, verbose=False)
    X, y, s = ld.load_all_subjects(n_subjects=70)
    res['beta_2s'] = evaluate(X.astype(np.float64), y, s, bf)
    print('beta', {k: round(np.mean(v) * 100, 2) for k, v in res['beta_2s'].items()})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
