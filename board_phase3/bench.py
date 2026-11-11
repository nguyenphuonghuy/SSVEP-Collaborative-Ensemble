"""
board_phase3/bench.py  (runs on Raspberry Pi 4 and ZCU104 PS)
==============================================================
Phase-3 latency benchmark requested by reviewers. One configuration per
process (the caller repeats processes). Modes:

  ensemble       3-branch probability averaging (deployed model)
  ensemble_dft   same, with the exact-DFT SSVEPFreqNet branch (post hoc)
  freqnet        SSVEPFreqNet alone
  fbcca_ref      reference FBCCA (QR of references recomputed per call;
                 identical algorithm to the paper's baseline)
  fbcca_cached   FBCCA with reference QR precomputed once per target and the
                 data QR computed once per sub-band per trial

Every iteration times the full path from a windowed EEG trial (numpy array,
9 x T) to the class label, with stages:
  NN:    prep (z-normalization + tensor) | infer (3 forwards + averaging + argmax)
  FBCCA: filt (5 zero-phase sub-band filters) | cca (correlations + argmax)

Thread count is fixed BEFORE numpy/torch import (OMP/OpenBLAS/MKL env vars and
torch.set_num_threads). Output: JSON with per-iteration latencies (ms).
"""
import os
import sys
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--mode', required=True,
                choices=['ensemble', 'ensemble_dft', 'freqnet', 'fbcca_ref', 'fbcca_cached'])
ap.add_argument('--n_samples', type=int, required=True)
ap.add_argument('--threads', type=int, default=1)
ap.add_argument('--warmup', type=int, default=20)
ap.add_argument('--iters', type=int, default=200)
ap.add_argument('--data', default=None, help='npz with X (n,9,T) real trials; random if absent')
ap.add_argument('--out', required=True)
A = ap.parse_args()
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[v] = str(A.threads)

import json, time, platform, resource  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal, linalg  # noqa: E402

FS, NCH, NCLS = 250.0, 9, 40
FREQS = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2,
                  8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4,
                  8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6,
                  8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8])


# ---------------------------------------------------------------- FBCCA (paper's reference implementation)
def canonical_correlation(X, Y):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    Qx, Rx = linalg.qr(X, mode='economic')
    Qy, Ry = linalg.qr(Y, mode='economic')
    if np.linalg.matrix_rank(Rx) < Rx.shape[1] or np.linalg.matrix_rank(Ry) < Ry.shape[1]:
        pass
    try:
        s = linalg.svd(Qx.T @ Qy, compute_uv=False)
        return float(s[0]) if len(s) > 0 else 0.0
    except np.linalg.LinAlgError:
        return 0.0


def build_reference_signals(freqs, n_harmonics, n_samples, sfreq):
    t = np.arange(n_samples) / sfreq
    refs = np.zeros((len(freqs), n_samples, 2 * n_harmonics))
    for i, f in enumerate(freqs):
        for h in range(1, n_harmonics + 1):
            refs[i, :, 2 * (h - 1)] = np.sin(2 * np.pi * h * f * t)
            refs[i, :, 2 * (h - 1) + 1] = np.cos(2 * np.pi * h * f * t)
    return refs


def make_filter_bank(n_subbands, sfreq, low_base=8.0, high=90.0):
    nyq = sfreq / 2.0
    out = []
    for k in range(1, n_subbands + 1):
        low = max(k * low_base - 2, 1.0)
        high_k = min(high, nyq - 1)
        if low >= high_k:
            low = high_k - 5
        out.append(signal.butter(4, [low / nyq, high_k / nyq], btype='band', output='sos'))
    return out


W_SB = np.array([n ** (-1.25) + 0.25 for n in range(1, 6)])


def fbcca_ref(trial, refs, fb, stages):
    scores = np.zeros(len(refs))
    t_f = t_c = 0.0
    for k, sos in enumerate(fb):
        t0 = time.perf_counter()
        X = signal.sosfiltfilt(sos, trial, axis=-1).T
        t1 = time.perf_counter()
        for n in range(len(refs)):
            scores[n] += W_SB[k] * canonical_correlation(X, refs[n]) ** 2
        t_c += time.perf_counter() - t1
        t_f += t1 - t0
    stages['filt'].append(t_f * 1e3); stages['cca'].append(t_c * 1e3)
    return int(np.argmax(scores))


def make_cached(refs):
    Q = []
    for R in refs:
        Rc = R - R.mean(0, keepdims=True)
        Q.append(linalg.qr(Rc, mode='economic')[0])
    return np.stack(Q)                                   # (40, T, 10)


def fbcca_cached(trial, Qy, fb, stages):
    scores = np.zeros(len(Qy))
    t_f = t_c = 0.0
    for k, sos in enumerate(fb):
        t0 = time.perf_counter()
        X = signal.sosfiltfilt(sos, trial, axis=-1).T
        t1 = time.perf_counter()
        Xc = X - X.mean(0, keepdims=True)
        Qx = linalg.qr(Xc, mode='economic')[0]           # once per sub-band
        M = np.einsum('tc,ntr->ncr', Qx, Qy)             # (40, 9, 10)
        s = np.linalg.svd(M, compute_uv=False)[:, 0]
        scores += W_SB[k] * s ** 2
        t_c += time.perf_counter() - t1
        t_f += t1 - t0
    stages['filt'].append(t_f * 1e3); stages['cca'].append(t_c * 1e3)
    return int(np.argmax(scores))


def summarize(v):
    v = np.asarray(v)
    return dict(mean=float(v.mean()), sd=float(v.std()), median=float(np.median(v)),
                p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)), max=float(v.max()), n=int(len(v)))


def main():
    T = A.n_samples
    if A.data and os.path.exists(A.data):
        X = np.load(A.data)['X'].astype(np.float32)
        assert X.shape[-1] == T, (X.shape, T)
    else:
        X = np.random.RandomState(0).randn(64, NCH, T).astype(np.float32)
    n_trials = len(X)
    stages = {}
    lat, preds = [], []

    if A.mode.startswith('fbcca'):
        fb = make_filter_bank(5, FS)
        refs = build_reference_signals(FREQS, 5, T, FS)
        stages = {'filt': [], 'cca': []}
        if A.mode == 'fbcca_ref':
            fn = lambda x: fbcca_ref(x.astype(np.float64), refs, fb, stages)
        else:
            Qy = make_cached(refs)
            fn = lambda x: fbcca_cached(x.astype(np.float64), Qy, fb, stages)
        for i in range(A.warmup):
            fn(X[i % n_trials])
        for k in stages: stages[k].clear()
        for i in range(A.iters):
            x = X[i % n_trials]
            t0 = time.perf_counter(); p = fn(x); lat.append((time.perf_counter() - t0) * 1e3); preds.append(p)
        versions = dict(numpy=np.__version__, scipy=__import__('scipy').__version__)
    else:
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(A.threads)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from models.extractors.extractors import build_extractor
        names = {'ensemble': ['temporal_cnn', 'freq_domain_cnn', 'spatial_temporal_cnn'],
                 'ensemble_dft': ['temporal_cnn', 'freq_domain_cnn_dft', 'spatial_temporal_cnn'],
                 'freqnet': ['freq_domain_cnn']}[A.mode]
        models = [build_extractor(n, NCH, T, NCLS).eval() for n in names]
        stages = {'prep': [], 'infer': []}

        def fn(x):
            t0 = time.perf_counter()
            z = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-8)
            xt = torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32)).unsqueeze(0)
            t1 = time.perf_counter()
            with torch.no_grad():
                p = sum(F.softmax(m(xt), -1) for m in models) / len(models)
                y = int(p.argmax(-1))
            t2 = time.perf_counter()
            stages['prep'].append((t1 - t0) * 1e3); stages['infer'].append((t2 - t1) * 1e3)
            return y
        for i in range(A.warmup):
            fn(X[i % n_trials])
        for k in stages: stages[k].clear()
        for i in range(A.iters):
            x = X[i % n_trials]
            t0 = time.perf_counter(); p = fn(x); lat.append((time.perf_counter() - t0) * 1e3); preds.append(p)
        versions = dict(torch=torch.__version__, numpy=np.__version__)

    res = dict(mode=A.mode, n_samples=T, threads=A.threads, iters=A.iters, warmup=A.warmup,
               data='real' if A.data and os.path.exists(A.data) else 'random',
               platform=platform.platform(), machine=platform.machine(), versions=versions,
               total=summarize(lat), stages={k: summarize(v) for k, v in stages.items()},
               latencies_ms=[round(v, 4) for v in lat],
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
    try:
        res['governor'] = open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor').read().strip()
        res['cur_freq_khz'] = int(open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq').read())
    except OSError:
        pass
    json.dump(res, open(A.out, 'w'))
    t = res['total']
    print(f"{A.mode} T={T} thr={A.threads}: median {t['median']:.2f} p95 {t['p95']:.2f} p99 {t['p99']:.2f} mean {t['mean']:.2f} ms")


if __name__ == '__main__':
    main()
