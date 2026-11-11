"""
board_phase3/aggregate.py  (runs on the HPC)
============================================
Aggregates Phase-3 board runs fetched into results/phase3/<board>/runs/ and
the per-trial agreement file agree_results.json.

Latency: for each (mode, window, threads), the per-process mean, median, p95,
p99 and max are averaged across processes (mean +/- SD across processes).
Agreement: board predictions/probabilities of the deployed averaging model vs
the GPU probabilities saved by evaluation/branch_combos.py (bm_5s_probs.npz).
"""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent / 'results'


def lat(board):
    groups = defaultdict(list)
    for f in glob.glob(str(R / 'phase3' / board / 'runs' / '*.json')):
        m = re.match(r'(.+)_(\d)s_t(\d)_r(\d+)\.json', Path(f).name)
        d = json.load(open(f))
        groups[(m.group(1), m.group(2) + 's', int(m.group(3)))].append(d)
    out = {}
    for k, runs in sorted(groups.items()):
        agg = {}
        for stat in ('mean', 'median', 'p95', 'p99', 'max'):
            v = np.array([r['total'][stat] for r in runs])
            agg[stat] = [float(v.mean()), float(v.std())]
        for st in runs[0]['stages']:
            v = np.array([r['stages'][st]['mean'] for r in runs])
            agg['stage_' + st] = [float(v.mean()), float(v.std())]
        agg['rss_mb'] = [float(np.mean([r['peak_rss_mb'] for r in runs])), float(np.std([r['peak_rss_mb'] for r in runs]))]
        agg['n_runs'] = len(runs)
        agg['versions'] = runs[0]['versions']
        agg['governor'] = runs[0].get('governor')
        out['|'.join(map(str, k))] = agg
    return out


def agreement(board):
    f = R / 'phase3' / board / 'agree_results.json'
    if not f.exists():
        return None
    a = json.load(open(f))
    z = np.load(R / 'branch_combos' / 'bm_5s_probs.npz')
    P, y, subj = z['P'].mean(0), z['y'], z['subj']
    out = {}
    for s, d in a.items():
        m = subj == int(s)
        gpu_pred = P[m].argmax(1)
        bp = np.array(d['pred'])
        out[s] = dict(n=int(m.sum()), agree=int((bp == gpu_pred).sum()),
                      max_abs_prob_diff=float(np.abs(np.array(d['prob']) - P[m]).max()),
                      acc_board=d['acc'], acc_gpu=float((gpu_pred == y[m]).mean()))
    return out


def main():
    res = {}
    for board in ('pi4', 'zcu104'):
        if not (R / 'phase3' / board).exists():
            continue
        res[board] = dict(latency=lat(board), agreement=agreement(board))
        print(f'===== {board}')
        L = res[board]['latency']
        for k, v in L.items():
            mode, w, thr = k.split('|')
            st = ' '.join(f"{s[6:]}={v[s][0]:.2f}" for s in v if s.startswith('stage_'))
            print(f"{mode:13s} {w} t{thr}: mean {v['mean'][0]:8.2f}+/-{v['mean'][1]:.2f}  median {v['median'][0]:8.2f}  "
                  f"p95 {v['p95'][0]:8.2f}  p99 {v['p99'][0]:8.2f}  max {v['max'][0]:8.2f}  rss {v['rss_mb'][0]:.0f}MB  [{st}] n={v['n_runs']}")
        for w in ('1s', '2s', '4s', '5s'):
            e = {t: L.get(f'ensemble|{w}|{t}') for t in (1, 4)}
            for fb_mode in ('fbcca_cached', 'fbcca_ref'):
                for t in (1, 4):
                    fb = L.get(f'{fb_mode}|{w}|{t}')
                    if fb and e[t]:
                        print(f'  speedup {w} {fb_mode} vs ensemble at {t} thread(s): {fb["mean"][0] / e[t]["mean"][0]:.1f}x')
        if res[board]['agreement']:
            for s, d in res[board]['agreement'].items():
                print(f'  agreement S{s}: {d["agree"]}/{d["n"]}, max |dp| {d["max_abs_prob_diff"]:.2e}, acc board {d["acc_board"]:.4f} gpu {d["acc_gpu"]:.4f}')
    json.dump(res, open(R / 'phase3_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
