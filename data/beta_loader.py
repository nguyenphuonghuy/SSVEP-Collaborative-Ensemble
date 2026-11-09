"""
data/beta_loader.py
====================
Loader cho BETA Dataset (Tsinghua) — Task 6: Cross-dataset validation.

BETA specs:
  - 70 subjects (S1.mat ... S70.mat)
  - 64 channels, 250 Hz
  - 40 targets (cùng frequencies với Benchmark)
  - 4 blocks × 3s trial = 750 samples
  - Format: data['EEG'] shape (64, 750, 4, 40)
            channels × samples × blocks × targets

Cross-dataset protocol:
  Train: Benchmark (35 subjects, fully trained)
  Test : BETA (70 subjects, zero-shot hoặc fine-tune nhỏ)
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

# 0-indexed positions of Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2, verified
# directly against BETA's own embedded channel table (data['suppl_info']['chan'],
# identical across subjects). The former value [45,51,52,53,54,55,58,59,60]
# was WRONG: it selects P3, P8, PO7, PO5, PO3, POz, PO8, CB1, O1 instead.
SELECTED_CHANNEL_INDICES = [47, 53, 54, 55, 56, 57, 60, 61, 62]
# SELECTED_CHANNEL_INDICES = [45, 51, 52, 53, 54, 55, 58, 59, 60]  # old, wrong (kept for pilot A/B revert)
CHANNEL_NAMES = ['Pz', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'O1', 'Oz', 'O2']

# BETA (Liu et al. 2020): S1-S15 have a 3s/750-sample trial buffer (0.5s
# pre-stimulus cue + 2.0s stimulation + 0.5s rest); S16-S70 have a
# 4s/1000-sample buffer (0.5s cue + 3.0s stimulation + 0.5s rest). Both
# groups share the same 0.5s pre-stimulus cue, so stimulus onset is at
# sample index 125 (0.5s @ 250Hz) in either buffer. t_start/t_end are
# seconds relative to this TRUE stimulus onset, matching Benchmark's
# convention (data/benchmark_loader.py) and how the paper describes the
# window. The per-subject buffer length (750 or 1000 samples) is read
# from the loaded array itself in load_subject(), not hardcoded, so a
# window request is capped correctly for whichever group a subject is in.
BETA_SFREQ = 250
BETA_STIM_ONSET_SAMPLE = 125   # 0.5 s pre-stimulus cue @ 250 Hz
BETA_TRIAL_SAMPLES = 750   # legacy fallback (S1-S15 buffer length); kept only
                           # for the t_end=None ("use full trial") case below


class BETADataset:
    """
    Loader cho BETA Dataset.

    Usage:
        loader = BETADataset('data/raw/beta')
        X, y = loader.load_subject(subject_id=1)
        # X: (n_trials, n_channels, n_samples)
        # y: (n_trials,) labels 0-39
    """

    def __init__(
        self,
        data_path: str,
        sfreq: int = 250,
        t_start: float = 0.14,    # visual latency
        t_end: Optional[float] = None,  # None = dùng hết trial
        selected_channels: Optional[List[int]] = None,
        bandpass: Optional[Tuple[float, float]] = (6.0, 90.0),
        verbose: bool = True
    ):
        self.data_path = Path(data_path)
        self.sfreq = sfreq
        self.t_start = t_start
        self.t_end = t_end
        self.selected_channels = selected_channels or SELECTED_CHANNEL_INDICES
        self.bandpass = bandpass
        self.verbose = verbose

        # Onset-relative sample index; the per-subject buffer length (750
        # for S1-S15, 1000 for S16-S70) is applied as the upper cap inside
        # load_subject(), where the actual array shape is known.
        self.sample_start = BETA_STIM_ONSET_SAMPLE + int(round(t_start * sfreq))
        if t_end is None:
            self._sample_end_uncapped = None  # capped to each subject's own buffer end
        else:
            self._sample_end_uncapped = BETA_STIM_ONSET_SAMPLE + int(round(t_end * sfreq))

        # Nominal n_samples assuming the shorter (S1-S15, 750-sample) buffer;
        # load_subject() recomputes the exact slice per subject.
        nominal_end = self._sample_end_uncapped if self._sample_end_uncapped is not None else BETA_TRIAL_SAMPLES
        self.sample_end = min(nominal_end, BETA_TRIAL_SAMPLES)
        self.n_samples = self.sample_end - self.sample_start
        self.n_channels = len(self.selected_channels)
        self.n_targets = 40
        self.n_blocks = 4

        mat_files = list(self.data_path.glob("S*.mat"))
        if self.verbose:
            logger.info(f"BETA: tìm thấy {len(mat_files)} subjects tại {self.data_path}")

    def load_subject(
        self,
        subject_id: int,
        blocks: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data của 1 subject từ BETA.

        Returns:
            X: (n_trials, n_channels, n_samples)
            y: (n_trials,) labels 0-39
        """
        mat_path = self.data_path / f"S{subject_id}.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {mat_path}")

        mat = sio.loadmat(str(mat_path), simplify_cells=True)
        # EEG shape: (64, n_buf, 4, 40) — channels x samples x blocks x targets.
        # n_buf is 750 (3s buffer, S1-S15) or 1000 (4s buffer, S16-S70); both
        # groups share the same 0.5s pre-stimulus cue, so the onset-relative
        # sample_start/sample_end computed in __init__ apply to either. Cap
        # defensively against this subject's actual buffer length so a
        # too-long window request truncates instead of raising an index
        # error, and warn if that happens (it would break the assumption of
        # a uniform n_samples across subjects).
        raw = mat['data']['EEG']
        subject_buffer_len = raw.shape[1]
        sample_end = min(self.sample_end, subject_buffer_len)
        if sample_end - self.sample_start != self.n_samples and self.verbose:
            logger.warning(
                f"BETA subject {subject_id}: requested window [{self.sample_start},"
                f" {self.sample_end}) exceeds this subject's buffer length "
                f"({subject_buffer_len}); truncated to {sample_end - self.sample_start} samples "
                f"instead of the nominal {self.n_samples}."
            )

        if blocks is None:
            blocks = list(range(self.n_blocks))

        # Select channels và time window
        # raw: (64, n_buf, 4, 40)
        n_samp = sample_end - self.sample_start
        data = raw[self.selected_channels, :, :, :]   # (9, n_buf, 4, 40)
        data = data[:, self.sample_start:sample_end, :, :]  # (9, n_samp, 4, 40)
        data = data[:, :, blocks, :]                  # (9, n_samp, n_b, 40)

        n_b = len(blocks)
        # Reshape → (n_trials, n_channels, n_samples)
        # (9, n_samp, n_b, 40) → transpose → (40, n_b, 9, n_samp) → reshape
        data = data.transpose(3, 2, 0, 1)             # (40, n_b, 9, n_samp)
        data = data.reshape(-1, self.n_channels, n_samp)  # (160, 9, n_samp)

        # Labels
        targets = np.arange(self.n_targets)
        labels = np.repeat(targets, n_b)              # (160,)

        if self.bandpass:
            data = self._bandpass_filter(data)
        data = self._normalize(data)

        if self.verbose:
            logger.info(f"BETA Subject {subject_id}: X={data.shape}, y={labels.shape}")

        return data.astype(np.float32), labels.astype(np.int64)

    def load_all_subjects(
        self,
        n_subjects: int = 70
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load toàn bộ BETA dataset."""
        all_X, all_y, all_subj = [], [], []

        for subj_id in range(1, n_subjects + 1):
            try:
                X, y = self.load_subject(subj_id)
                all_X.append(X)
                all_y.append(y)
                all_subj.append(np.full(len(y), subj_id, dtype=np.int64))
            except FileNotFoundError:
                logger.warning(f"Skip subject {subj_id}")

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        subject_ids = np.concatenate(all_subj, axis=0)
        logger.info(f"BETA loaded: {len(np.unique(subject_ids))} subjects, X={X.shape}")
        return X, y, subject_ids

    def _bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        from scipy.signal import butter, filtfilt
        low, high = self.bandpass
        nyq = self.sfreq / 2.0
        b, a = butter(4, [low / nyq, high / nyq], btype='band')
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            for c in range(data.shape[1]):
                filtered[i, c] = filtfilt(b, a, data[i, c])
        return filtered

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        mean = data.mean(axis=-1, keepdims=True)
        std  = data.std(axis=-1, keepdims=True) + 1e-8
        return (data - mean) / std
