"""
latency_benchmark_pi.py
========================
Do latency inference cua mo hinh ensemble chinh (3 extractor:
temporal_cnn + freq_domain_cnn + spatial_temporal_cnn, cua so 5.44s)
tren Raspberry Pi 4 (CPU, aarch64).

Latency inference chi phu thuoc kien truc (so phep tinh / do thi tensor),
KHONG phu thuoc gia tri trong so da hoc -> dung trong so khoi tao ngau
nhien la hop le cho muc dich do latency (khong danh gia accuracy o day).

Do rieng tung extractor (batch=1, tuan tu - dung kich ban trien khai thuc
te tren thiet bi edge don luong) + tong 3 extractor + buoc average-softmax,
voi ca hai case so luong thread CPU: 1 (single-core) va full (4 core).

Usage:
    python latency_benchmark_pi.py --out results.json
"""
import sys
import json
import time
import resource
import argparse
import platform
import statistics as stats
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from models.extractors.extractors import build_extractor

N_CHANNELS = 9
N_CLASSES = 40
EXTRACTORS = ['temporal_cnn', 'freq_domain_cnn', 'spatial_temporal_cnn']

N_WARMUP = 20
N_TRIALS = 200


def percentile(values, p):
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def bench_module(model, x, n_warmup=N_WARMUP, n_trials=N_TRIALS):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)  # ms
    return times


def summarize(times):
    return {
        'mean_ms': stats.mean(times),
        'std_ms': stats.stdev(times) if len(times) > 1 else 0.0,
        'median_ms': stats.median(times),
        'p95_ms': percentile(times, 95),
        'p99_ms': percentile(times, 99),
        'min_ms': min(times),
        'max_ms': max(times),
        'n_trials': len(times),
    }


def run_config(n_threads: int, n_samples: int, extractors=None, do_ensemble=True):
    torch.set_num_threads(n_threads)
    torch.manual_seed(0)
    extractors = extractors or EXTRACTORS

    x = torch.randn(1, N_CHANNELS, n_samples)

    per_ext = {}
    models = []
    for name in extractors:
        m = build_extractor(name, N_CHANNELS, n_samples, N_CLASSES)
        n_params = sum(p.numel() for p in m.parameters())
        times = bench_module(m, x)
        per_ext[name] = {**summarize(times), 'n_params': n_params}
        models.append(m)
        print(f"  [{n_threads}thr] {name}: mean={per_ext[name]['mean_ms']:.3f}ms "
              f"p95={per_ext[name]['p95_ms']:.3f}ms n_params={n_params}")

    if not do_ensemble:
        return {'per_extractor': per_ext, 'ensemble': None}

    # Full ensemble pipeline: 3 forward pass (tuan tu, nhu tren thiet bi
    # edge don luong) + average softmax
    for m in models:
        m.eval()
    with torch.no_grad():
        for _ in range(N_WARMUP):
            probs = [F.softmax(m(x), dim=-1) for m in models]
            torch.stack(probs, dim=0).mean(dim=0)
        # CPU utilization: so sanh CPU-time (tong qua tat ca luong) voi
        # wall-time cua dung N_TRIALS lan lap ensemble. Ty le nay xap xi so
        # "loi" duoc su dung trung binh (vd 1.0 = tuong duong 1 loi full-tai).
        cpu_t0 = time.process_time()
        wall_t0 = time.perf_counter()
        ens_times = []
        for _ in range(N_TRIALS):
            t0 = time.perf_counter()
            probs = [F.softmax(m(x), dim=-1) for m in models]
            avg = torch.stack(probs, dim=0).mean(dim=0)
            ens_times.append((time.perf_counter() - t0) * 1000.0)
        cpu_elapsed = time.process_time() - cpu_t0
        wall_elapsed = time.perf_counter() - wall_t0
    ens_summary = summarize(ens_times)
    ens_summary['cpu_cores_used_avg'] = round(cpu_elapsed / wall_elapsed, 3) if wall_elapsed > 0 else None
    print(f"  [{n_threads}thr] ENSEMBLE ({len(extractors)} extractor + avg-softmax): "
          f"mean={ens_summary['mean_ms']:.3f}ms p95={ens_summary['p95_ms']:.3f}ms "
          f"cpu_cores_used_avg={ens_summary['cpu_cores_used_avg']}")

    return {'per_extractor': per_ext, 'ensemble': ens_summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='pi4_latency_results.json')
    parser.add_argument('--n_samples', type=int, default=1360,
                         help='So mau input @ 250Hz (1360=5.44s, 1000=4s, 500=2s, 250=1s)')
    parser.add_argument('--extractors', type=str, default=None,
                         help='Danh sach ten extractor cach nhau boi dau phay '
                              '(mac dinh: 3 extractor cua ensemble chinh)')
    parser.add_argument('--no_ensemble', action='store_true',
                         help='Bo qua buoc do ensemble gop (dung khi benchmark '
                              'cac extractor doc lap, khong phai mot to hop thuc te)')
    args = parser.parse_args()
    extractors = args.extractors.split(',') if args.extractors else None

    result = {
        'device': 'Raspberry Pi 4',
        'platform': platform.platform(),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'input_shape': [1, N_CHANNELS, args.n_samples],
        'window_s': args.n_samples / 250.0,
        'note': 'Trong so khoi tao ngau nhien (seed=0) - latency inference '
                'khong phu thuoc gia tri trong so da hoc, chi phu thuoc kien truc.',
        'configs': {},
    }

    for n_threads in [1, 4]:
        print(f"=== num_threads={n_threads} ===")
        result['configs'][f'{n_threads}_threads'] = run_config(
            n_threads, args.n_samples, extractors=extractors,
            do_ensemble=not args.no_ensemble
        )

    # Peak RSS (bo nho vat ly) cua ca tien trinh tinh tu luc khoi dong -
    # bao gom nap PyTorch + xay 3 model + toan bo cac lan chay o tren.
    # ru_maxrss tren Linux tra ve don vi KB.
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['peak_rss_mb'] = round(peak_rss_kb / 1024, 2)
    print(f"Peak RSS memory: {result['peak_rss_mb']:.2f} MB")

    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
