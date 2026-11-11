"""Ensemble with the DFT-based SSVEPFreqNet branch substituted for the FFT one
(reuses per-trial temporal/spatial probabilities saved by branch_combos.py).
Compares T+Fdft+S with the original T+F+S and with same-pipeline FBCCA."""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent.parent)); sys.path.insert(0, str(Path(__file__).parent))
from models.extractors.extractors import build_extractor
from data.benchmark_loader import BenchmarkDataset
from data.beta_loader import BETADataset
from phase1_stats import paired, fbcca
R = Path('results')
cfg = {'1s': ('bm_1s', 1.14, 'benchmark'), '2s': ('bm_2s', 2.14, 'benchmark'),
       '4s': ('bm_4s', 4.14, 'benchmark'), 'beta': ('beta_2s', 2.14, 'beta')}
out = {}
for w, (tag, t_end, ds) in cfg.items():
    z = np.load(R / 'branch_combos' / f'{tag}_probs.npz')
    P, y, subj = z['P'], z['y'], z['subj']
    if ds == 'beta':
        X, y2, s2 = BETADataset(data_path='data/raw/beta', t_start=0.14, t_end=t_end, verbose=False).load_all_subjects(70)
    else:
        X, y2, s2 = BenchmarkDataset(data_path='data/raw/benchmark', t_start=0.14, t_end=t_end, verbose=False).load_all_subjects(35)
    assert np.array_equal(y, y2) and np.array_equal(subj, s2)
    Pd = np.zeros_like(P[1])
    ck = R / 'phase2' / f'freq_domain_cnn_dft_{w}'
    for fold, s in enumerate(np.unique(subj), start=1):
        m = subj == s
        net = build_extractor('freq_domain_cnn_dft', X.shape[1], X.shape[2], 40)
        net.load_state_dict(torch.load(ck / f'freq_domain_cnn_dft_fold{fold}.pt', map_location='cpu')); net.eval()
        with torch.no_grad():
            Pd[m] = F.softmax(net(torch.from_numpy(X[m]).float()), -1).numpy()
    acc = lambda pred: np.array([(pred[subj == s] == y[subj == s]).mean() for s in np.unique(subj)])
    fd = acc(Pd.argmax(-1)); old = acc(P.mean(0).argmax(-1))
    new = acc(np.stack([P[0], Pd, P[2]]).mean(0).argmax(-1))
    fb = np.array(fbcca(w if w != 'beta' else 'beta'))
    r_old, r_fb = paired(new, old), paired(new, fb)
    out[w] = dict(fdft=fd.mean()*100, ens_old=old.mean()*100, ens_dft=new.mean()*100, ens_dft_sd=new.std()*100,
                  new_vs_old=r_old, new_vs_fbcca=r_fb, per_subject=new.tolist())
    print(f"{w}: Fdft {fd.mean()*100:.2f} | ens old {old.mean()*100:.2f} -> new {new.mean()*100:.2f}+/-{new.std()*100:.2f} "
          f"({r_old['diff']:+.2f}, p={r_old['p']:.3g}, CI [{r_old['ci'][0]:.2f},{r_old['ci'][1]:.2f}]) | vs FBCCA {r_fb['mean_b']:.2f}: "
          f"{r_fb['diff']:+.2f}, p={r_fb['p']:.3g}, CI [{r_fb['ci'][0]:.2f},{r_fb['ci'][1]:.2f}], +{r_fb['n_better']}/-{r_fb['n_worse']}", flush=True)
json.dump(out, open(R / 'phase2' / 'dft_ensemble.json', 'w'), indent=1, default=float)
