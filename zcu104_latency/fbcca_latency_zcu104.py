"""
fbcca_latency_pi.py
=====================
Do latency phan loai 1 trial cua FBCCA (filter-bank canonical correlation
analysis, baseline co dien khong hoc - doi chieu voi ensemble hoc duoc,
novelty ablation) tren Raspberry Pi 4.

Thuat toan/hang so port nguyen ven tu evaluation/fbcca_baseline.py (cung
cong thuc eq:fbcca, a=1.25/b=0.25, 40 target x 5 harmonic, 5 sub-band Chen
et al. 2015) - khong sua doi de dam bao phan anh dung chi phi cua phuong
phap dang dung de bao cao ket qua accuracy trong bai.

Du lieu dau vao la tin hieu tong hop (random) vi latency cua CCA/filtfilt
chi phu thuoc kich thuoc du lieu (n_channels, n_samples), khong phu thuoc
gia tri tin hieu thuc.

Usage:
    python fbcca_latency_pi.py --n_samples 1360 --out results.json
"""
import sys
import json
import time
import resource
import argparse
import platform
import statistics as stats

import numpy as np
from scipy import signal, linalg

N_CHANNELS = 9
SFREQ = 250.0
N_HARMONICS = 5
N_SUBBANDS = 5
N_WARMUP = 3
N_TRIALS = 20

TARGET_FREQS = np.array([
    8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2,
    8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4,
    8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6,
    8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8,
])


def canonical_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    Qx, Rx = linalg.qr(X, mode='economic')
    Qy, Ry = linalg.qr(Y, mode='economic')
    try:
        M = Qx.T @ Qy
        s = linalg.svd(M, compute_uv=False)
        return float(s[0]) if len(s) > 0 else 0.0
    except np.linalg.LinAlgError:
        return 0.0


def build_reference_signals(freqs, n_harmonics, n_samples, sfreq):
    t = np.arange(n_samples) / sfreq
    n_targets = len(freqs)
    refs = np.zeros((n_targets, n_samples, 2 * n_harmonics))
    for i, f in enumerate(freqs):
        for h in range(1, n_harmonics + 1):
            refs[i, :, 2 * (h - 1)] = np.sin(2 * np.pi * f * h * t)
            refs[i, :, 2 * (h - 1) + 1] = np.cos(2 * np.pi * f * h * t)
    return refs


def make_filter_bank(n_subbands, sfreq, low_base=8.0, high=90.0):
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


def fbcca_classify_trial(trial, ref_signals, filter_bank, a=1.25, b=0.25):
    n_subbands = len(filter_bank)
    n_targets = ref_signals.shape[0]
    weights = np.array([n ** (-a) + b for n in range(1, n_subbands + 1)])
    scores = np.zeros(n_targets)
    for sb_idx, sos in enumerate(filter_bank):
        filtered = signal.sosfiltfilt(sos, trial, axis=-1)
        X = filtered.T
        for t_idx in range(n_targets):
            rho = canonical_correlation(X, ref_signals[t_idx])
            scores[t_idx] += weights[sb_idx] * (rho ** 2)
    return int(np.argmax(scores))


def percentile(values, p):
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def summarize(times):
    return {
        'mean_ms': stats.mean(times), 'std_ms': stats.stdev(times) if len(times) > 1 else 0.0,
        'median_ms': stats.median(times), 'p95_ms': percentile(times, 95),
        'p99_ms': percentile(times, 99), 'min_ms': min(times), 'max_ms': max(times),
        'n_trials': len(times),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_samples', type=int, default=1360)
    ap.add_argument('--out', type=str, default='fbcca_latency_results.json')
    args = ap.parse_args()

    rng = np.random.RandomState(0)
    trial = rng.randn(N_CHANNELS, args.n_samples) * 10e-6  # thang do EEG dien hinh (V)

    ref_signals = build_reference_signals(TARGET_FREQS, N_HARMONICS, args.n_samples, SFREQ)
    filter_bank = make_filter_bank(N_SUBBANDS, SFREQ)

    for _ in range(N_WARMUP):
        fbcca_classify_trial(trial, ref_signals, filter_bank)

    cpu_t0 = time.process_time()
    wall_t0 = time.perf_counter()
    times = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        fbcca_classify_trial(trial, ref_signals, filter_bank)
        times.append((time.perf_counter() - t0) * 1000.0)
    cpu_elapsed = time.process_time() - cpu_t0
    wall_elapsed = time.perf_counter() - wall_t0

    ens_summary = summarize(times)
    ens_summary['cpu_cores_used_avg'] = round(cpu_elapsed / wall_elapsed, 3) if wall_elapsed > 0 else None
    print(f"  FBCCA (n_samples={args.n_samples}, {args.n_samples/SFREQ:.2f}s): "
          f"mean={ens_summary['mean_ms']:.3f}ms p95={ens_summary['p95_ms']:.3f}ms "
          f"cpu_cores_used_avg={ens_summary['cpu_cores_used_avg']}")

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mb = round(peak_rss_kb / 1024, 2)
    print(f"Peak RSS memory: {peak_rss_mb:.2f} MB")

    result = {
        'device': 'ZCU104 (PYNQ, Cortex-A53)',
        'platform': platform.platform(),
        'python': platform.python_version(),
        'method': 'FBCCA',
        'input_shape': [N_CHANNELS, args.n_samples],
        'window_s': args.n_samples / SFREQ,
        'n_targets': len(TARGET_FREQS), 'n_harmonics': N_HARMONICS, 'n_subbands': N_SUBBANDS,
        'note': 'Tin hieu dau vao tong hop (random) - latency CCA/filtfilt chi phu '
                'thuoc kich thuoc du lieu, khong phu thuoc gia tri tin hieu thuc.',
        # Schema tuong thich voi aggregate_runs.py (dung chung script gop N lan chay)
        'configs': {'default_threads': {'per_extractor': {}, 'ensemble': ens_summary}},
        'peak_rss_mb': peak_rss_mb,
    }
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == '__main__':
    main()
