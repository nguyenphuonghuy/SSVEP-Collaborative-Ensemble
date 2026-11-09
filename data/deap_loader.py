"""
data/deap_loader.py
====================
Loader cho DEAP (Koelstra et al. 2012), dùng làm testbed thứ hai (ngoài
SSVEP) để kiểm chứng tính tổng quát của khung chưng cất cộng tác — cùng
interface với BenchmarkDataset/BETADataset (load_all_subjects -> X, y,
subject_ids) để tái sử dụng nguyên vẹn pipeline huấn luyện/đánh giá hiện có
(train_extractors.py, collaborative_ensemble.py).

Định dạng file gốc (`data_preprocessed_python/sXX.dat`, pickle):
  - 'data':   (40 trial, 40 kênh, 8064 mẫu @128Hz) — 32 kênh EEG đầu +
              8 kênh ngoại vi (EOG/EMG/GSR/hô hấp/nhiệt độ) ở cuối, đã lọc
              thông dải 4.0-45.0Hz, hạ mẫu 128Hz, loại nhiễu EOG sẵn từ nhà
              phát hành (không cần lọc lại — xem BenchmarkDataset, cũng chỉ
              normalize không filter).
  - 'labels': (40 trial, 4) = [valence, arousal, dominance, liking], thang 1-9.

Tiền xử lý: cắt bỏ 3s baseline đầu mỗi trial (chuẩn của tài liệu DEAP), chỉ
giữ 32 kênh EEG, cắt cửa sổ cố định không chồng lấn, nhị phân hoá nhãn theo
ngưỡng giữa thang đo (4.5), chuẩn hoá z-score theo trục thời gian mỗi cửa sổ
(cùng quy ước với BenchmarkDataset).
"""
import pickle
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

SFREQ = 128.0
N_EEG_CHANNELS = 32          # 32 kenh dau, 8 kenh ngoai vi bi bo
BASELINE_SAMPLES = int(3 * SFREQ)   # 3s baseline dau moi trial
GROUND_TRUTH_THRESHOLD = 4.5        # thang 1-9, nguong giua


class DEAPDataset:
    def __init__(self, data_path, label_type='valence', window_sec=4.0,
                 overlap=0.0, verbose=True):
        """
        Args:
            data_path: thu muc chua truc tiep s01.dat...s32.dat
            label_type: 'valence' hoac 'arousal'
            window_sec: do dai cua so (giay), khong chong lap mac dinh
            overlap: ty le chong lap giua cac cua so lien tiep [0,1)
        """
        assert label_type in ('valence', 'arousal')
        self.data_path = Path(data_path)
        self.label_type = label_type
        self.label_idx = 0 if label_type == 'valence' else 1
        self.window_sec = window_sec
        self.n_samples = int(window_sec * SFREQ)
        self.step = int(self.n_samples * (1 - overlap))
        self.verbose = verbose
        self.n_channels = N_EEG_CHANNELS
        self.n_classes = 2
        dat_files = sorted(self.data_path.glob("s*.dat"))
        if verbose:
            logger.info(f"Found {len(dat_files)} DEAP subjects, "
                        f"window={window_sec}s ({self.n_samples} samples), "
                        f"label={label_type}")

    def load_subject(self, subject_id):
        dat_path = self.data_path / f"s{subject_id:02d}.dat"
        if not dat_path.exists():
            raise FileNotFoundError(f"Not found: {dat_path}")
        with open(dat_path, 'rb') as f:
            d = pickle.load(f, encoding='latin-1')
        raw = d['data'][:, :N_EEG_CHANNELS, :]          # (40, 32, 8064)
        raw = raw[:, :, BASELINE_SAMPLES:]               # bo 3s baseline -> (40,32,7680)
        scores = d['labels'][:, self.label_idx]          # (40,) thang 1-9
        binary_labels = (scores >= GROUND_TRUTH_THRESHOLD).astype(np.int64)

        n_trials, n_ch, n_t = raw.shape
        n_windows = (n_t - self.n_samples) // self.step + 1

        all_X, all_y = [], []
        for tr in range(n_trials):
            for w in range(n_windows):
                s0 = w * self.step
                seg = raw[tr, :, s0:s0 + self.n_samples]
                all_X.append(seg)
                all_y.append(binary_labels[tr])
        X = np.stack(all_X).astype(np.float32)            # (n_trials*n_windows, 32, n_samples)
        y = np.array(all_y, dtype=np.int64)

        mean = X.mean(axis=-1, keepdims=True)
        std = X.std(axis=-1, keepdims=True) + 1e-8
        X = (X - mean) / std

        if self.verbose:
            logger.info(f"Subject {subject_id}: X={X.shape} "
                        f"(pos={y.sum()}/{len(y)}, {y.mean()*100:.1f}%)")
        return X, y

    def load_all_subjects(self, n_subjects=32):
        all_X, all_y, all_s = [], [], []
        for sid in range(1, n_subjects + 1):
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
        logger.info(f"Loaded {len(np.unique(s))} DEAP subjects: X={X.shape}, "
                    f"label={self.label_type}, pos_rate={y.mean()*100:.1f}%")
        return X, y, s
