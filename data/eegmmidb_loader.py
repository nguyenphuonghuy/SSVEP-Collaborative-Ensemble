"""
data/eegmmidb_loader.py
=========================
Loader cho PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB), dùng
làm testbed thứ hai (ngoài SSVEP) để kiểm chứng tính tổng quát của khung
chưng cất cộng tác — cùng interface với BenchmarkDataset/BETADataset/
DEAPDataset (load_all_subjects -> X, y, subject_ids).

Nguồn: PhysioNet EEGMMIDB (Goldberger et al. 2000; Schalk et al. 2004),
103 subject, 64 kênh EEG hệ 10-10, sfreq=160Hz. Chỉ dùng 6 run vận động
TƯỞNG TƯỢNG (motor imagery), bỏ qua các run vận động thật và baseline:
  - R04, R08, R12: T1 = tưởng tượng nắm tay TRÁI, T2 = tưởng tượng nắm tay PHẢI
  - R06, R10, R14: T1 = tưởng tượng nắm CẢ HAI TAY, T2 = tưởng tượng đá CẢ HAI CHÂN
  (T0 = nghỉ, không dùng, giữ đúng bài toán 4 lớp chuẩn của tài liệu MI-BCI
  trên chính bộ dữ liệu này).

Nhãn: 0=left_fist, 1=right_fist, 2=both_fists, 3=both_feet.

Kênh: chỉ giữ 9 kênh vùng cảm giác--vận động (FC3, FCz, FC4, C3, Cz, C4,
CP3, CPz, CP4 — lưới 3x3 đối xứng quanh đường giữa, bao phủ vỏ não
tiền--trung tâm và trung tâm--sau), tương tự cách chọn 9 kênh vùng chẩm
cho SSVEP (Mục Tiền xử lý dữ liệu trong bài báo).

Tiền xử lý: lọc thông dải 8-30Hz (băng mu+beta, chứa tín hiệu ERD/ERS đặc
trưng của vận động tưởng tượng — khác dải 6-90Hz dùng cho SSVEP vì MI
không có cấu trúc hài cần giữ lại các bậc hài cao), cắt cửa sổ cố định
[0, 4]s tính từ thời điểm bắt đầu cue (chuẩn trong tài liệu MI trên chính
bộ dữ liệu này), chuẩn hoá z-score theo trục thời gian mỗi cửa sổ (cùng
quy ước với BenchmarkDataset/DEAPDataset).

Đã quét toàn bộ 103 subject x 6 run (618 file): sfreq=160Hz, 64 kênh,
30 event/run đồng nhất cho mọi file, không phát hiện bất thường (không có
subject lỗi cần loại như P39 của AMIGOS).
"""
import warnings
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

SFREQ = 160.0
CHANNELS = ["Fc3.", "Fcz.", "Fc4.", "C3..", "Cz..", "C4..", "Cp3.", "Cpz.", "Cp4."]
BANDPASS = (8.0, 30.0)
TMIN, TMAX = 0.0, 4.0  # giay, tinh tu thoi diem cue

# (run, T1_label, T2_label)
RUN_LABEL_MAP = [
    ("R04", 0, 1), ("R08", 0, 1), ("R12", 0, 1),   # left_fist / right_fist
    ("R06", 2, 3), ("R10", 2, 3), ("R14", 2, 3),   # both_fists / both_feet
]
LABEL_NAMES = ["left_fist", "right_fist", "both_fists", "both_feet"]


class EEGMMIDBDataset:
    def __init__(self, data_path, verbose=True):
        self.data_path = Path(data_path)
        self.verbose = verbose
        self.n_samples = int(round((TMAX - TMIN) * SFREQ))
        self.n_channels = len(CHANNELS)
        self.n_classes = 4
        if verbose:
            logger.info(f"EEGMMIDB: window=[{TMIN},{TMAX}]s ({self.n_samples} samples), "
                        f"9 kenh cam giac--van dong, 4 lop {LABEL_NAMES}")

    def load_subject(self, subject_id):
        import mne
        mne.set_log_level('ERROR')
        sdir = self.data_path / f"S{subject_id:03d}"
        if not sdir.exists():
            raise FileNotFoundError(f"Not found: {sdir}")

        all_X, all_y = [], []
        for run, t1_label, t2_label in RUN_LABEL_MAP:
            fpath = sdir / f"S{subject_id:03d}{run}.edf"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.read_raw_edf(fpath, preload=True, verbose='ERROR')
            raw.pick(CHANNELS)
            raw.filter(BANDPASS[0], BANDPASS[1], fir_design='firwin', verbose='ERROR')

            events, event_id = mne.events_from_annotations(raw, verbose='ERROR')
            epochs = mne.Epochs(
                raw, events, event_id={'T1': event_id['T1'], 'T2': event_id['T2']},
                tmin=TMIN, tmax=TMAX, baseline=None, preload=True,
                verbose='ERROR'
            )
            data = epochs.get_data(copy=True)[:, :, :self.n_samples]  # (n_trial, 9, n_samples)
            labels_raw = epochs.events[:, -1]
            t1_code, t2_code = event_id['T1'], event_id['T2']
            labels = np.where(labels_raw == t1_code, t1_label, t2_label)

            all_X.append(data.astype(np.float32))
            all_y.append(labels.astype(np.int64))

        X = np.concatenate(all_X)
        y = np.concatenate(all_y)

        mean = X.mean(axis=-1, keepdims=True)
        std = X.std(axis=-1, keepdims=True) + 1e-8
        X = (X - mean) / std

        if self.verbose:
            counts = np.bincount(y, minlength=4)
            logger.info(f"Subject {subject_id}: X={X.shape}, "
                        f"phan bo lop={dict(zip(LABEL_NAMES, counts.tolist()))}")
        return X, y

    def load_all_subjects(self, n_subjects=103):
        all_X, all_y, all_s = [], [], []
        n_loaded = 0
        for sid in range(1, 110):
            if n_loaded >= n_subjects:
                break
            try:
                X, y = self.load_subject(sid)
            except FileNotFoundError:
                continue
            all_X.append(X)
            all_y.append(y)
            all_s.append(np.full(len(y), sid, dtype=np.int64))
            n_loaded += 1
        X = np.concatenate(all_X)
        y = np.concatenate(all_y)
        s = np.concatenate(all_s)
        logger.info(f"Loaded {len(np.unique(s))} EEGMMIDB subjects: X={X.shape}, "
                    f"phan bo lop={dict(zip(LABEL_NAMES, np.bincount(y, minlength=4).tolist()))}")
        return X, y, s
