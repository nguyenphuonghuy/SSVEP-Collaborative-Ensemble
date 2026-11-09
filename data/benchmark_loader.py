
import numpy as np
import scipy.io as sio
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

BENCHMARK_FREQUENCIES = np.array([
    8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,
    8.2,9.2,10.2,11.2,12.2,13.2,14.2,15.2,
    8.4,9.4,10.4,11.4,12.4,13.4,14.4,15.4,
    8.6,9.6,10.6,11.6,12.6,13.6,14.6,15.6,
    8.8,9.8,10.8,11.8,12.8,13.8,14.8,15.8,
])

# 0-indexed positions of Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2 in the
# official 64-channel Benchmark montage (Wang et al. 2017; verified against
# 64-channels.loc from the Tsinghua group server and against BETA's own
# embedded channel table, which uses the same montage family). The former
# value [45,51,52,53,54,55,58,59,60] was WRONG: it actually selects
# P3, P8, PO7, PO5, PO3, POz, PO8, CB1, O1 (off by up to 2 positions),
# missing Pz, PO4, PO6, Oz and O2 -- the strongest SSVEP channels.
SELECTED_CHANNEL_INDICES = [47,53,54,55,56,57,60,61,62]
# SELECTED_CHANNEL_INDICES = [45,51,52,53,54,55,58,59,60]  # old, wrong (kept for pilot A/B revert)

# True acquisition rate of the Benchmark dataset (Wang et al. 2017): 250 Hz.
# Each trial buffer is 1500 samples = 6.0 s = 0.5 s pre-stimulus cue +
# 5.0 s stimulation + 0.5 s post-stimulus rest. t_start/t_end below are
# seconds relative to TRUE STIMULUS ONSET (not to the start of the raw
# 6 s buffer), matching how the paper describes the window ("extracted
# from the stimulation onset after compensating for latency").
BENCHMARK_TRUE_SFREQ = 250
BENCHMARK_STIM_ONSET_SAMPLE = 125   # 0.5 s pre-stimulus cue @ 250 Hz
BENCHMARK_BUFFER_SAMPLES = 1500     # 6.0 s @ 250 Hz


class BenchmarkDataset:
    def __init__(self, data_path, sfreq=BENCHMARK_TRUE_SFREQ, t_start=0.0, t_end=1.5,
                 selected_channels=None, bandpass=None, verbose=True):
        self.data_path = Path(data_path)
        self.sfreq = sfreq
        self.t_start = t_start
        self.t_end = t_end
        self.selected_channels = selected_channels or SELECTED_CHANNEL_INDICES
        self.bandpass = bandpass
        self.verbose = verbose
        # sample indices are onset-relative: index 0 of the raw buffer is
        # the start of the pre-stimulus cue, not stimulus onset, so the
        # onset offset must be added explicitly.
        self.sample_start = BENCHMARK_STIM_ONSET_SAMPLE + int(round(t_start * sfreq))
        self.sample_end   = min(BENCHMARK_STIM_ONSET_SAMPLE + int(round(t_end * sfreq)),
                                 BENCHMARK_BUFFER_SAMPLES)
        self.n_samples    = self.sample_end - self.sample_start
        self.n_channels   = len(self.selected_channels)
        self.n_targets    = 40
        self.n_blocks     = 6
        mat_files = list(self.data_path.glob("S*.mat"))
        logger.info(f"Found {len(mat_files)} subjects, n_samples={self.n_samples}")

    def load_subject(self, subject_id, blocks=None):
        mat_path = self.data_path / f"S{subject_id}.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"Not found: {mat_path}")
        mat = sio.loadmat(str(mat_path))
        raw = mat["data"]  # (64, 1500, 40, 6)
        if blocks is None:
            blocks = list(range(self.n_blocks))
        data = raw[self.selected_channels, :, :, :]
        data = data[:, self.sample_start:self.sample_end, :, :]
        data = data[:, :, :, blocks]
        n_b  = len(blocks)
        data = data.transpose(2,3,0,1).reshape(-1, self.n_channels, self.n_samples)
        labels = np.repeat(np.arange(self.n_targets), n_b)
        # Chi normalize, KHONG filter
        mean = data.mean(axis=-1, keepdims=True)
        std  = data.std(axis=-1,  keepdims=True) + 1e-8
        data = (data - mean) / std
        if self.verbose:
            logger.info(f"Subject {subject_id}: X={data.shape}")
        return data.astype(np.float32), labels.astype(np.int64)

    def load_all_subjects(self, n_subjects=35):
        all_X, all_y, all_s = [], [], []
        for sid in range(1, n_subjects+1):
            try:
                X, y = self.load_subject(sid)
                all_X.append(X)
                all_y.append(y)
                all_s.append(np.full(len(y), sid, dtype=np.int64))
            except FileNotFoundError:
                pass
        X = np.concatenate(all_X)
        y = np.concatenate(all_y)
        s = np.concatenate(all_s)
        logger.info(f"Loaded {len(np.unique(s))} subjects: X={X.shape}")
        return X, y, s

def create_dummy_data(n_subjects=3, n_channels=9, n_samples=1500,
                      n_targets=40, n_blocks=6):
    np.random.seed(42)
    n_t = n_targets * n_blocks
    all_X, all_y, all_s = [], [], []
    for s in range(1, n_subjects+1):
        t = np.linspace(0, n_samples/1000.0, n_samples)
        X = np.random.randn(n_t, n_channels, n_samples) * 0.1
        for i in range(n_t):
            freq = BENCHMARK_FREQUENCIES[i % n_targets]
            X[i] += np.sin(2*np.pi*freq*t)[None,:]
        y = np.tile(np.arange(n_targets), n_blocks)
        all_X.append(X); all_y.append(y)
        all_s.append(np.full(n_t, s))
    return (np.concatenate(all_X).astype(np.float32),
            np.concatenate(all_y).astype(np.int64),
            np.concatenate(all_s).astype(np.int64))
