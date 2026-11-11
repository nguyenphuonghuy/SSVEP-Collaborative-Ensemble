"""
evaluation/phase2_stats.py
==========================
Collects Phase 2 reviewer-control runs (results/phase2/<extractor>_<window>/)
and compares each with SSVEPFreqNet at the same window (paired over subjects,
difference = variant minus SSVEPFreqNet, CI from t(n-1)).

Also reports the training-free harmonic score (results/phase2/harmonic_score.json).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from phase1_stats import paired  # noqa: E402

R = Path(__file__).resolve().parent.parent / 'results'
TAG = {'1s': 'bm_1s', '2s': 'bm_2s', '4s': 'bm_4s', '5s': 'bm_5s', 'beta': 'beta_2s'}


def freqnet(window):
    c = json.load(open(R / 'branch_combos' / f'{TAG[window]}_combos.json'))
    return np.array(c['per_subject_acc']['freq_domain_cnn'])


def run_acc(d):
    fs = sorted(glob.glob(str(d / '*_loso_results.json')))
    if not fs:
        return None, None
    r = json.load(open(fs[-1]))
    fo = sorted(r['folds'], key=lambda f: f['subject'])
    return np.array([f['best_acc'] for f in fo]), r.get('n_subjects', len(fo))


def main():
    out = {}
    for d in sorted((R / 'phase2').glob('*_*')):
        if not d.is_dir():
            continue
        name, window = d.name.rsplit('_', 1)
        acc, n = run_acc(d)
        if acc is None:
            print(f'{d.name:40s} not finished ({len(list(d.glob("*.done.json")))} folds done)')
            continue
        f = freqnet(window)
        r = paired(acc, f) if len(acc) == len(f) else None
        out[d.name] = dict(mean=float(acc.mean() * 100), sd=float(acc.std() * 100), n=len(acc), vs_freqnet=r)
        s = '' if r is None else (f"vs SSVEPFreqNet {r['diff']:+.2f} pp, t={r['t']:.2f}, p={r['p']:.4g}, "
                                  f"CI [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}], +{r['n_better']}/-{r['n_worse']}")
        print(f'{d.name:40s} {acc.mean()*100:6.2f} +/- {acc.std()*100:5.2f}  {s}')
    hs = R / 'phase2' / 'harmonic_score.json'
    if hs.exists():
        h = json.load(open(hs))
        out['harmonic_score'] = {k: {m: float(np.mean(v) * 100) for m, v in d.items()} for k, d in h.items()}
        for k, d in out['harmonic_score'].items():
            print('harmonic score', k, {m: round(v, 2) for m, v in d.items()})
    json.dump(out, open(R / 'phase2_stats.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
