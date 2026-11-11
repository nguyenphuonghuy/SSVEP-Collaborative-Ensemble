"""
evaluation/phase1_stats.py
==========================
Reviewer-driven statistics computed from existing per-subject results
(no training). Run after evaluation/branch_combos.py and fbcca_beta.py.

  1. Branch combinations (1/2/3 branches) per window/dataset, with paired
     tests of every combination against SSVEPFreqNet alone.
  2. Ensemble vs. SSVEPFreqNet across the three end-to-end seeds.
  3. Ensemble vs. same-pipeline FBCCA per window (and BETA if available).
  4. Per-subject ITR (mean +/- SD) next to ITR from mean accuracy.
  5. Collaborative-minus-averaging over seeds 1001/2002 with subject as unit.

Convention: differences are A minus B as named; CI from t(n-1).
Output: results/phase1_stats.json and a printed summary.
"""
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

R = Path(__file__).resolve().parent.parent / 'results'
FULL = 'temporal_cnn+freq_domain_cnn+spatial_temporal_cnn'
FREQ = 'freq_domain_cnn'
SHORT = {'temporal_cnn': 'T', 'freq_domain_cnn': 'F', 'spatial_temporal_cnn': 'S'}


def paired(a, b):
    a, b = np.asarray(a) * 100, np.asarray(b) * 100
    d = a - b
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    t = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(a, b, zero_method='wilcox').pvalue
    except ValueError:
        w = float('nan')
    return dict(mean_a=a.mean(), mean_b=b.mean(), diff=d.mean(), t=float(t.statistic), p=float(t.pvalue),
                wilcoxon_p=float(w), d=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float('nan'),
                ci=[d.mean() - tc * se, d.mean() + tc * se], n_better=int((d > 0).sum()), n_worse=int((d < 0).sum()),
                n=n)


def itr(P, T, N=40):
    P = min(max(P, 1e-12), 1.0)
    if P <= 1.0 / N:
        return 0.0
    b = math.log2(N) + P * math.log2(P) + ((1 - P) * math.log2((1 - P) / (N - 1)) if P < 1 else 0)
    return b * 60 / T


def combos(tag):
    f = R / 'branch_combos' / f'{tag}_combos.json'
    return json.load(open(f))['per_subject_acc'] if f.exists() else None


def fbcca(window):
    pat = {'1s': 'fbcca_1s_fixedwindow_round', '2s': 'fbcca_2s_fixedwindow_round',
           '4s': 'fbcca_4s_fixedwindow_round', '5s': 'fbcca_fixedwindow_chfix',
           'beta': 'fbcca_beta_fixedwindow_round'}[window]
    fs = sorted(glob.glob(str(R / pat / '*.json')))
    return json.load(open(fs[-1]))['per_subject_acc'] if fs else None


def collab_minus_avg(path):
    d = json.load(open(path))
    fo = sorted(d['folds'], key=lambda f: f['subject'])
    return np.array([f['best_acc'] for f in fo]), np.array([f['baseline_acc'] for f in fo])


def main():
    out = {}
    print('== 1. Branch combinations (mean %, and paired vs SSVEPFreqNet)')
    out['combos'] = {}
    for tag in ['bm_1s', 'bm_2s', 'bm_4s', 'bm_5s', 'bm_5s_seed1001', 'bm_5s_seed2002', 'beta_2s']:
        c = combos(tag)
        if c is None:
            print(f'  {tag}: missing'); continue
        out['combos'][tag] = {}
        print(f'  {tag}')
        for k, v in c.items():
            lab = '+'.join(SHORT[x] for x in k.split('+'))
            r = paired(v, c[FREQ]) if k != FREQ else None
            out['combos'][tag][lab] = dict(mean=float(np.mean(v) * 100), sd=float(np.std(v) * 100), vs_F=r)
            extra = '' if r is None else f"  vs F: {r['diff']:+.2f} pp, t={r['t']:.2f}, p={r['p']:.4f}, CI [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}], +{r['n_better']}/-{r['n_worse']}"
            print(f'    {lab:6s} {np.mean(v)*100:6.2f} +/- {np.std(v)*100:5.2f}{extra}')

    print('== 2. Seeds (5.0 s Benchmark)')
    seeds = {s: combos(t) for s, t in [('42', 'bm_5s'), ('1001', 'bm_5s_seed1001'), ('2002', 'bm_5s_seed2002')]}
    if all(seeds.values()):
        ens = np.array([np.mean(seeds[s][FULL]) * 100 for s in seeds])
        fr = np.array([np.mean(seeds[s][FREQ]) * 100 for s in seeds])
        ens_subj = np.mean([seeds[s][FULL] for s in seeds], 0)
        fr_subj = np.mean([seeds[s][FREQ] for s in seeds], 0)
        r = paired(ens_subj, fr_subj)
        out['seeds'] = dict(ensemble=ens.tolist(), freq=fr.tolist(), subj_avg_paired=r)
        print(f'  ensemble per seed {np.round(ens,2)} mean {ens.mean():.2f} +/- {ens.std(ddof=1):.2f}')
        print(f'  SSVEPFreqNet per seed {np.round(fr,2)} mean {fr.mean():.2f} +/- {fr.std(ddof=1):.2f}')
        print(f"  seed-averaged per subject: ens - F = {r['diff']:+.2f} pp, p={r['p']:.4f}, CI [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]")

    print('== 3. Ensemble vs FBCCA (ensemble minus FBCCA)')
    out['vs_fbcca'] = {}
    for w, tag in [('1s', 'bm_1s'), ('2s', 'bm_2s'), ('4s', 'bm_4s'), ('5s', 'bm_5s'), ('beta', 'beta_2s')]:
        c, fb = combos(tag), fbcca(w)
        if c is None or fb is None:
            print(f'  {w}: missing'); continue
        r = paired(c[FULL], fb)
        out['vs_fbcca'][w] = r
        print(f"  {w}: ens {r['mean_a']:.2f} vs FBCCA {r['mean_b']:.2f}: {r['diff']:+.2f} pp, t={r['t']:.2f}, p={r['p']:.4g}, "
              f"Wilcoxon p={r['wilcoxon_p']:.4g}, CI [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}], +{r['n_better']}/-{r['n_worse']}")

    print('== 4. ITR: from mean accuracy vs per-subject mean +/- SD')
    out['itr'] = {}
    L = {'bm_1s': 1.0, 'bm_2s': 2.0, 'bm_4s': 4.0, 'bm_5s': 5.0, 'beta_2s': 2.0}
    for tag, w in L.items():
        c = combos(tag)
        fbk = {'bm_1s': '1s', 'bm_2s': '2s', 'bm_4s': '4s', 'bm_5s': '5s', 'beta_2s': 'beta'}[tag]
        for name, acc in [('ensemble', c[FULL] if c else None), ('freq', c[FREQ] if c else None), ('fbcca', fbcca(fbk))]:
            if acc is None:
                continue
            T = w + 0.5
            per = [itr(a, T) for a in acc]
            out['itr'][f'{tag}_{name}'] = dict(from_mean=itr(float(np.mean(acc)), T), subj_mean=float(np.mean(per)), subj_sd=float(np.std(per)))
            print(f'  {tag:8s} {name:8s} from-mean {itr(float(np.mean(acc)), T):6.2f} | per-subject {np.mean(per):6.2f} +/- {np.std(per):6.2f}')

    print('== 5. Collaborative minus averaging, seeds 1001/2002, subject as unit')
    f1 = sorted(glob.glob(str(R / 'ensemble_full_fixedwindow_round_seed1001' / 'collaborative_loso_*seed1001.json')))
    f2 = sorted(glob.glob(str(R / 'ensemble_full_fixedwindow_round_seed2002' / 'collaborative_loso_*seed2002.json')))
    if f1 and f2:
        c1, b1 = collab_minus_avg(f1[-1])
        c2, b2 = collab_minus_avg(f2[-1])
        r = paired((c1 + c2) / 2, (b1 + b2) / 2)
        pooled = paired(np.r_[c1, c2], np.r_[b1, b2])
        out['collab_seeds'] = dict(subject_unit=r, pooled_70=pooled)
        print(f"  subject-averaged (n=35): {r['diff']:+.2f} pp, t={r['t']:.2f}, p={r['p']:.3f}, d={r['d']:.2f}, CI [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]")
        print(f"  (old pooled n=70:        {pooled['diff']:+.2f} pp, p={pooled['p']:.3f}, CI [{pooled['ci'][0]:.2f}, {pooled['ci'][1]:.2f}])")

    json.dump(out, open(R / 'phase1_stats.json', 'w'), indent=1, default=float)
    print('saved', R / 'phase1_stats.json')


if __name__ == '__main__':
    main()
