"""NOTE: the accuracy_verify files dated 2026-08-16 predate the window/channel fix and do
not match current checkpoints; rerun pi4_latency/accuracy_verify_pi.py with current
collab or averaging checkpoints and save per-trial predictions before using this.
Trial-level agreement between Pi 4 (ARM) predictions stored in
results/latency_pi4/accuracy_verify/ and predictions recomputed on the
x86/GPU host from the same collaborative checkpoints (5.0 s window)."""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.extractors.extractors import build_extractor
from data.benchmark_loader import BenchmarkDataset
B = ['temporal_cnn', 'freq_domain_cnn', 'spatial_temporal_cnn']
ck = Path('results/ensemble_full_fixedwindow_round')
ld = BenchmarkDataset(data_path='data/raw/benchmark', t_start=0.14, t_end=5.14, verbose=False)
out = {}
import argparse
ap = argparse.ArgumentParser(); ap.add_argument('--dir', default='results/latency_pi4/accverify_round'); ap.add_argument('--device', default='cpu'); ap.add_argument('--out', default='results/latency_pi4/trial_agreement_round.json'); A = ap.parse_args()
dev = torch.device(A.device)
for f in sorted(Path(A.dir).glob('acc*_s*.json')):
    d = json.load(open(f)); s = d['subject']
    X, y = ld.load_subject(s)
    assert np.array_equal(np.asarray(d['labels']), y), f'label order mismatch S{s}'
    P = []
    with torch.no_grad():
        for b in B:
            m = build_extractor(b, X.shape[1], X.shape[2], 40)
            m.load_state_dict(torch.load(ck / f'{b}_collab_fold{s}.pt', map_location='cpu')); m.to(dev).eval()
            P.append(F.softmax(m(torch.from_numpy(X).float().to(dev)), -1).cpu())
    host = torch.stack(P).mean(0).argmax(1).numpy()
    pi = np.asarray(d['predictions'])
    out[s] = dict(n=len(y), agree=int((host == pi).sum()), acc_host=float((host == y).mean()), acc_pi=float((pi == y).mean()), device=str(dev))
    print(s, out[s])
json.dump(out, open(A.out, 'w'), indent=1)
