"""
aggregate_runs.py
===================
Gop N lan chay doc lap (moi lan la 1 tien trinh python rieng, tu
latency_benchmark_pi.py) thanh thong ke lien-tien-trinh (across-run):
mean, std, 95% CI cua "mean_ms" tung extractor + ensemble, cho ca hai
cau hinh so thread (1 va 4).

Ly do can lam vay (khong chi dua vao N=200 lan lap TRONG 1 tien trinh):
bien dong giua cac tien trinh phan anh nhieu he dieu hanh (scheduling),
trang thai cache/bo nho, va throttling nhiet cua Pi 4 - thu do khong
tien trinh don le nao bat duoc het.

Usage:
    python aggregate_runs.py --pattern "run_*.json" --out summary.json
"""
import sys
import json
import glob
import argparse
import statistics as stats


def agg(values):
    n = len(values)
    mean = stats.mean(values)
    sd = stats.stdev(values) if n > 1 else 0.0
    se = sd / (n ** 0.5) if n > 1 else 0.0
    t_crit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
              7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)
    return {
        'mean_ms': mean, 'std_ms': sd, 'n_runs': n,
        'ci95_ms': [mean - t_crit * se, mean + t_crit * se],
        'per_run_mean_ms': values,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', type=str, default='run_*.json')
    ap.add_argument('--out', type=str, default='pi4_latency_summary_Nruns.json')
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"Khong tim thay file khop '{args.pattern}'")
        sys.exit(1)
    print(f"Gop {len(files)} lan chay: {files}")

    runs = [json.load(open(f)) for f in files]
    thread_keys = list(runs[0]['configs'].keys())
    ext_names = list(runs[0]['configs'][thread_keys[0]]['per_extractor'].keys())

    summary = {
        'n_runs': len(files), 'source_files': files,
        'device': runs[0]['device'], 'torch': runs[0].get('torch', runs[0].get('method', 'N/A')),
        'input_shape': runs[0]['input_shape'], 'window_s': runs[0]['window_s'],
        'configs': {},
    }

    for tk in thread_keys:
        summary['configs'][tk] = {'per_extractor': {}, 'ensemble': None}
        for ext in ext_names:
            vals = [r['configs'][tk]['per_extractor'][ext]['mean_ms'] for r in runs]
            summary['configs'][tk]['per_extractor'][ext] = agg(vals)
        ens_vals = [r['configs'][tk]['ensemble']['mean_ms'] for r in runs]
        summary['configs'][tk]['ensemble'] = agg(ens_vals)

        # CPU utilization (so loi trung binh duoc su dung) - chi co o cac
        # ban script da them tracking; bo qua neu file cu chua co truong nay.
        cpu_vals = [r['configs'][tk]['ensemble'].get('cpu_cores_used_avg')
                    for r in runs if r['configs'][tk]['ensemble'].get('cpu_cores_used_avg') is not None]
        if cpu_vals:
            summary['configs'][tk]['cpu_cores_used_avg'] = agg(cpu_vals)

        print(f"\n=== {tk} ===")
        for ext in ext_names:
            a = summary['configs'][tk]['per_extractor'][ext]
            print(f"  {ext}: {a['mean_ms']:.3f} ± {a['std_ms']:.3f} ms "
                  f"(CI95 [{a['ci95_ms'][0]:.3f}, {a['ci95_ms'][1]:.3f}])")
        a = summary['configs'][tk]['ensemble']
        print(f"  ENSEMBLE: {a['mean_ms']:.3f} ± {a['std_ms']:.3f} ms "
              f"(CI95 [{a['ci95_ms'][0]:.3f}, {a['ci95_ms'][1]:.3f}])")
        if cpu_vals:
            c = summary['configs'][tk]['cpu_cores_used_avg']
            print(f"  CPU cores used (avg): {c['mean_ms']:.3f} ± {c['std_ms']:.3f}")

    # Peak RSS (bo nho): lay tu cac file run (khong phu thuoc thread config,
    # phan anh peak cua ca tien trinh) neu co.
    rss_vals = [r.get('peak_rss_mb') for r in runs if r.get('peak_rss_mb') is not None]
    if rss_vals:
        summary['peak_rss_mb'] = agg(rss_vals)
        print(f"\nPeak RSS memory: {summary['peak_rss_mb']['mean_ms']:.1f} ± "
              f"{summary['peak_rss_mb']['std_ms']:.1f} MB")

    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
