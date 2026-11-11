"""
board_phase3/agree.py  (runs on the board)
==========================================
Per-trial predictions and averaged probabilities of the DEPLOYED model
(probability averaging of the independently trained branches) for the test
subjects exported in data/subject{S}_5s.npz, using the fold checkpoints in
ckpts/. Output is compared on the HPC with the GPU probabilities.
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.extractors.extractors import build_extractor

ap = argparse.ArgumentParser()
ap.add_argument('--subjects', default='5,9,13,14,35')
ap.add_argument('--out', default='agree_results.json')
A = ap.parse_args()
torch.set_num_threads(1)
B = ['temporal_cnn', 'freq_domain_cnn', 'spatial_temporal_cnn']
out = {}
for s in map(int, A.subjects.split(',')):
    d = np.load(f'data/subject{s}_5s.npz')
    X, y = torch.from_numpy(d['X']).float(), d['y']
    P = []
    with torch.no_grad():
        for b in B:
            m = build_extractor(b, X.shape[1], X.shape[2], 40)
            m.load_state_dict(torch.load(f'ckpts/{b}_fold{s}.pt', map_location='cpu'))
            m.eval()
            P.append(F.softmax(m(X), -1))
    p = torch.stack(P).mean(0).numpy()
    out[s] = dict(pred=p.argmax(1).tolist(), prob=p.astype(np.float64).round(8).tolist(),
                  acc=float((p.argmax(1) == y).mean()), torch=torch.__version__)
    print(s, out[s]['acc'])
json.dump(out, open(A.out, 'w'))
